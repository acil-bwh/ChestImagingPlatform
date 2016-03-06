#include "ShapeModelOptimizer.h"
#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include "ShapeModelVisualizer.h"

#include <vtkMath.h>

#include <climits>
#include <float.h>
#include <ctime>
#include <algorithm>

ShapeModelOptimizer::ShapeModelOptimizer( ShapeModel& shapeModel,
                                          ImageType::Pointer image )
: _shapeModel(shapeModel), _image(image)
{
  // make sure shape model already contains its initial transform at this time
  std::cout << "Initial transform: " << *_shapeModel.getTransform() << std::endl;
}

ShapeModelOptimizer::~ShapeModelOptimizer()
{
}

void
ShapeModelOptimizer::run( double maxSearchLength,
                          double sigma,
                          double decayFactor,
                          int maxIteration,
                          int numModes,
                          ShapeModelVisualizer& visualizer )
{
  prepareGradientImages( sigma );

  ImageType::SpacingType spacing = _image->GetSpacing();

  // examine gradient image for each model point along normal axis
  const double threshold = 0.1; // stopping criteria (maximum difference)
  const double minSearchLength = (1.0 - decayFactor) * maxSearchLength;

  std::cout << "=============== Fitting Parameters ===============" << std::endl;
  std::cout << "Starting search length: "        << maxSearchLength << std::endl;
  std::cout << "Gaussian gradient sigma: "       << sigma           << std::endl;
  std::cout << "Decay factor of search length: " << decayFactor     << std::endl;
  std::cout << "Max iterations: "                << maxIteration    << std::endl;
  std::cout << "Number of modes to use: "        << numModes        << std::endl;
  std::cout << "==================================================" << std::endl;

  unsigned int numPoints = _shapeModel.getNumberOfPoints();

  // to compute similarity transform between these two point sets
  vtkSmartPointer< vtkPoints > modelPoints = vtkSmartPointer< vtkPoints >::New();
  vtkSmartPointer< vtkPoints > targetImagePoints = vtkSmartPointer< vtkPoints >::New();

  modelPoints->SetNumberOfPoints( numPoints );
  targetImagePoints->SetNumberOfPoints( numPoints );

  // prepare transforms between image and model space
  vtkSmartPointer< vtkTransform > imageToModelTransform = vtkSmartPointer< vtkTransform >::New();
  vtkSmartPointer< vtkTransform > modelToImageTransform = vtkSmartPointer< vtkTransform >::New();

  // prepare normal generator to update normal in every iteration
  vtkSmartPointer< vtkPolyDataNormals > normalGenerator = vtkSmartPointer< vtkPolyDataNormals >::New();
  normalGenerator->ComputePointNormalsOn();
  normalGenerator->ComputeCellNormalsOff();
  normalGenerator->SplittingOff();

  // convert mean shape model points to vtk points
  // used for similarity transform from image space to model space
  vnlVectorToVTKPoints( _shapeModel.getMean(), modelPoints );

  // *******************************************************************
  // repeat pose & shape estimation
  // *******************************************************************
  enum FitMode { POSE_ONLY, SHAPE_ONLY, POSE_AND_SHAPE }; // internal for experiment
  FitMode fitMode = POSE_AND_SHAPE; // estimate pose first

  double p[3];
  double n[3];

  for (int m = 0; m < maxIteration; m++)
  {
    vtkSmartPointer< vtkPolyData > polydata = _shapeModel.getPolyData();
    vtkSmartPointer< vtkPoints > currentImagePoints = polydata->GetPoints();

    // polydata is not supposed to contain any normals at this time
    // create normals from scratch using updated polydata (from last iteration)
    normalGenerator->SetInputData( polydata );
    normalGenerator->Update();
    polydata = normalGenerator->GetOutput();
    vtkSmartPointer< vtkDataArray > normals = polydata->GetPointData()->GetNormals();

    assert( 0 != normals ); // normals newly generated

    double t = m / (double)(maxIteration - 1);
    double searchLength = (1 - t) * maxSearchLength + t * minSearchLength;

    double maxMaxMag = 0;

    std::vector< PointType > vecOrgPt; // temporary container
    std::vector< PointType > vecQt; // temporary container
    std::vector< double > vecMaxMag; // temporary container
    vnl_matrix< double > spacingMatrix( 3, 3 ); // to calculate a proper step search depending on normal
    spacingMatrix.set_identity();
    spacingMatrix( 0, 0 ) = spacing[0];
    spacingMatrix( 1, 1 ) = spacing[1];
    spacingMatrix( 2, 2 ) = spacing[2];

    for (unsigned int i = 0; i < numPoints; i++)
    {
      PointType pt, qt, lastInsidePt;
      currentImagePoints->GetPoint( i, p );
      normals->GetTuple( i, n );
      PointType orgPt( p );
      CovPixelType normal( n );

      qt = orgPt; // target image point

      // search for maximum gradient magnitude
      // only when gradient direction is same as normal direction
      double maxMag = 0; // maximum gradient magnitude for one sample
      IndexType pixelIndex, lastInsidePixelIndex;

      vnl_vector< double > nvec( n, 3 );
      double stepSearch = (spacingMatrix * nvec).magnitude(); // step search depends on the spacing values
      int N = (int)(searchLength / stepSearch / 2.0 + 0.5); // num steps per side

      for (int j = -N; j < N; j++) // explore image space along the normal
      {
        for (int k = 0; k < 3; k++)
        {
          pt[k] = orgPt[k] + j * stepSearch * n[k];
        }
        bool isInside = _gradientImage->TransformPhysicalPointToIndex( pt, pixelIndex );
        if (isInside)
        {
          lastInsidePt = pt;
          lastInsidePixelIndex = pixelIndex;
        }
        else
        {
          pt = lastInsidePt;
          pixelIndex = lastInsidePixelIndex;
        }
        CovPixelType gradDir = _gradientImage->GetPixel( pixelIndex );
        double mag = gradDir.GetNorm();
        if (normal * gradDir > 0 && mag > maxMag)
        {
          maxMag = mag;
          qt = pt;
        }
      }

      // maximum gradient magnitude for all samples
      maxMaxMag = std::max( maxMag, maxMaxMag );

      // store original point, target point, and maximum magnitude
      vecMaxMag.push_back( maxMag );
      vecOrgPt.push_back( orgPt );
      vecQt.push_back( qt );
    }

    // second phase: make the movement relative to the strength of the edge sigmal
    for (unsigned int i = 0; i < numPoints; i++)
    {
      PointType orgPt = vecOrgPt[i];
      PointType qt = vecQt[i];
      double maxMag = vecMaxMag[i];

      PointType::VectorType diff = qt - orgPt; // full offset
      PointType::VectorType::ValueType f = maxMag / std::max( maxMaxMag, 1.0 ); // fraction of full offset
      qt = orgPt + f * diff; // final target point
      targetImagePoints->SetPoint( i, qt[0], qt[1], qt[2] );
    }

    double avgDist = computeAverageDistanceBetweenTwoPointSets( currentImagePoints, targetImagePoints );
    std::cout << (fitMode == POSE_ONLY ? "[P] " : (fitMode == SHAPE_ONLY ? "[S] " : "[P+S] "));
    std::cout << "Iteration: " << m << " -----" << std::endl;
    std::cout << "   search range [" << -searchLength/2.0 << " " << searchLength/2.0 << "]" << std::endl;
    std::cout << "   average distance to target image points: " << avgDist << std::endl;
    std::cout << "   max magnitude amaong all sample directions: " << maxMaxMag << std::endl;

    if (fitMode != SHAPE_ONLY)
    {
      // find similarity transform between current points and updated points
      vtkSmartPointer< vtkLandmarkTransform > landmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
      landmarkTransform->SetSourceLandmarks( targetImagePoints );
      landmarkTransform->SetTargetLandmarks( modelPoints );
      landmarkTransform->SetModeToSimilarity(); // uniform scaling procrustes analysis
      landmarkTransform->Update();
      vtkSmartPointer< vtkMatrix4x4 > matrix = landmarkTransform->GetMatrix();
      vtkSmartPointer< vtkMatrix4x4 > inverseMatrix = landmarkTransform->GetLinearInverse()->GetMatrix();

      imageToModelTransform->SetMatrix( matrix );
      modelToImageTransform->SetMatrix( inverseMatrix );
    }

    // transform to model coordinate system
    vtkSmartPointer< vtkPoints > targetModelPoints = vtkSmartPointer< vtkPoints >::New();
    imageToModelTransform->TransformPoints( targetImagePoints, targetModelPoints );

    vnl_vector< double > targetModel( numPoints * 3 );
    vtkPointsToVNLVector( targetModelPoints, targetModel );

    vtkSmartPointer< vtkPoints > pointsBefore = vtkSmartPointer< vtkPoints >::New();
    pointsBefore->DeepCopy( _shapeModel.getPolyData()->GetPoints() );

    // ----------------------------------------------------
    // shape (pca coefficient) estimation
    // ----------------------------------------------------

    // estimate closest shape in PCA space
    if (fitMode != POSE_ONLY)
    {
      _shapeModel.project( targetModel, numModes );
    }

    // set the scale & pose to shape model
    _shapeModel.setTransform( modelToImageTransform );

    vtkSmartPointer< vtkPoints > pointsAfter = _shapeModel.getPolyData()->GetPoints();

    // measure differences in image space
    double maxDist;
    computeDistanceBetweenTwoPointSets( pointsBefore, pointsAfter, avgDist, maxDist );

    std::cout << "   maximum distance from previous iteration: " << maxDist << ", average: " << avgDist << std::endl;

    if (maxDist < threshold)
    {
      if (fitMode == POSE_ONLY)
      {
        // pose converged. switching to shape fitting
        fitMode = POSE_AND_SHAPE;
      }
      else
      {
        std::cout << "Stopping." << std::endl;
        break;
      }
    }
  } // for m (repeat shape & pose estimation)
  // *******************************************************************
}

void
ShapeModelOptimizer::prepareGradientImages( double sigma )
{
  GradientRecursiveGaussianImageFilterType::Pointer gradientFilter = GradientRecursiveGaussianImageFilterType::New();

  gradientFilter->SetInput( _image );
  gradientFilter->SetSigma( sigma );
  try
  {
    std::cout << "Running Gaussian gradient filter..." << std::endl;
    gradientFilter->Update();
    std::cout << "Done." << std::endl;
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
  _gradientImage = gradientFilter->GetOutput();
}
