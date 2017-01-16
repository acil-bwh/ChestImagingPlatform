#include "ShapeModelOptimizer.h"
#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include "ShapeModelVisualizer.h"
#include "ShapeModelUtils.h"

#include <vtkMath.h>

#include <climits>
#include <float.h>
#include <ctime>
#include <algorithm>

ShapeModelOptimizer::ShapeModelOptimizer( ShapeModel& shapeModel,
                                          ShapeModelImage& image )
: _shapeModel( shapeModel ),
  _image( image )
{
  // make sure shape model already contains its initial transform at this time
  std::cout << "Initial transform: " << *_shapeModel.getTransform()->GetMatrix() << std::endl;
}

ShapeModelOptimizer::~ShapeModelOptimizer()
{
}

void
ShapeModelOptimizer::run( const std::string& inputFileName,
                          double maxSearchLength,
                          double sigma,
                          double decayFactor,
                          int maxIteration,
                          int poseOnlyIteration,
                          int numModes,
                          bool verbose,
                          ShapeModelVisualizer& visualizer )
{
  beforeOptimization( sigma );

  double spacing[3];
  _image.getSpacing( spacing );

  // examine gradient image for each model point along normal axis
  const double threshold = 0.1; // stopping criteria (maximum difference)
  const double minSearchLength = (1.0 - decayFactor) * maxSearchLength;
  unsigned int maxNumModes = _shapeModel.getNumberOfModes();
  if (numModes < 0 || numModes > maxNumModes)
  {
    numModes = maxNumModes;
  }

  std::cout << "=============== Fitting Parameters ===============" << std::endl;
  std::cout << "Starting search length: "        << maxSearchLength << std::endl;
  std::cout << "Gaussian gradient sigma: "       << sigma           << std::endl;
  std::cout << "Decay factor of search length: " << decayFactor     << std::endl;
  std::cout << "Max iterations: "                << maxIteration    << std::endl;
  std::cout << "Pose-only iterations: "          << poseOnlyIteration << std::endl;
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
  FitMode fitMode = POSE_ONLY; // estimate pose first

  double p[3]; // source point
  double n[3]; // normal of source point
  double pn[3]; // perturbed normal
  bool perturb_normal = false; // currently disabled
  
  double avgDist = FLT_MAX;
  int numNotFound = 0;

  for (int m = 0; m < maxIteration; m++)
  {
    if (m >= poseOnlyIteration)
    {
      fitMode = POSE_AND_SHAPE;
    }

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

    std::vector< PointType > vecOrgPt; // temporary container
    std::vector< PointType > vecQt; // temporary container
    std::vector< double > vecMaxEval; // temporary container
    std::vector< double > vecMinEval; // temporary container
    std::vector< bool > vecFoundTarget; // temporary container
    vnl_matrix< double > spacingMatrix( 3, 3 ); // to calculate a proper step search depending on normal
    spacingMatrix.set_identity();
    spacingMatrix( 0, 0 ) = spacing[0];
    spacingMatrix( 1, 1 ) = spacing[1];
    spacingMatrix( 2, 2 ) = spacing[2];

    double maxMaxEval = -FLT_MAX; // maximum evaluation across all samples
    double minMinEval = FLT_MAX; // minimum evaluation across all samples
    numNotFound = 0;

    for (unsigned int i = 0; i < numPoints; i++)
    {
      PointType pt, qt, lastInsidePt;
      currentImagePoints->GetPoint( i, p );
      normals->GetTuple( i, n );
      PointType orgPt( p );
      CovPixelType normal( n );

      normal.Normalize(); // vtkPolyDataNormals does not always return normalized normals!
      n[0] = normal[0];
      n[1] = normal[1];
      n[2] = normal[2];

      qt = orgPt; // target image point

      // search for maximum gradient magnitude
      // only when gradient direction is same as normal direction
      IndexType pixelIndex, lastInsidePixelIndex;

      vnl_vector< double > nvec( n, 3 );
      double stepSearch = (spacingMatrix * nvec).magnitude(); // step search depends on the spacing values
      stepSearch /= getSamplingFactor(); // allows super(factor > 1)/under(factor < 1)-sampling
      const int N = (int)(searchLength / stepSearch / 2.0 + 0.5); // num steps per side
      const unsigned int maxTrial = (perturb_normal) ? ((m > poseOnlyIteration - 2) ? 1000 : 1) : 1;
      unsigned int trial = 0;
      bool foundTarget = false;
      double minEval;
      double maxEval; // maximum gradient magnitude for one sample
      //std::cout << "   step search: " << stepSearch << std::endl;

      srand(1); // initialize random number genrator to perturb normal
      
      while (!foundTarget && trial < maxTrial) // multiple trial with perturbed normals
      {
        int minj = INT_MAX;
        double prevEval = -1; // do not initialize with 0 (see ShapeModelOptimizerDT::updatePosition)
        minEval = FLT_MAX;
        maxEval = -FLT_MAX; // maximum gradient magnitude for one sample
        PointType prevPt = qt;
        pn[0] = n[0]; pn[1] = n[1]; pn[2] = n[2]; // reset to original normal
        perturbNormal( pn, trial );
        if (trial > 0)
        {
          //std::cout << trial << ": " << pn[0] << " " << pn[1] << " " << pn[2] << std::endl;
        }

        for (int j = -N; j < N; j++) // explore image space along the normal
        {
          for (int k = 0; k < 3; k++)
          {
            pt[k] = orgPt[k] + j * stepSearch * pn[k];
          }
          bool isInside = transformPhysicalPointToIndex( pt, pixelIndex );
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
          
          if (updatePosition( pt, pixelIndex, prevPt, prevEval,
                              qt, normal, maxEval, minEval,
                              j, minj ))
          {
            foundTarget = true;
          }
          prevPt = pt;
        } // for each search step
        trial += 1;
      } // while target was not found

      // maximum gradient magnitude for all samples
      maxMaxEval = std::max( maxEval, maxMaxEval );
      minMinEval = std::min( minEval, minMinEval );

      // store original point, target point, and maximum magnitude
      vecMaxEval.push_back( maxEval );
      vecMinEval.push_back( minEval );
      vecOrgPt.push_back( orgPt );
      if (false == foundTarget)
      {
        //std::cout << "Not found." << std::endl;
        numNotFound += 1;
      }
      vecQt.push_back( qt );
      vecFoundTarget.push_back( foundTarget );
    } // for each model point

    // second phase: make the movement relative to the strength of the edge signal
    double sumDist = 0;
    for (unsigned int i = 0; i < numPoints; i++)
    {
      PointType qt = determinePosition( i, vecOrgPt, vecQt, 
                                        vecMaxEval, vecMinEval, 
                                        maxMaxEval, minMinEval );
      targetImagePoints->SetPoint( i, qt[0], qt[1], qt[2] );

      // to compute average distance considering the case when the target point 
      // could not be found within search length
      if (vecFoundTarget[i])
      {
        sumDist += sqrt( vtkMath::Distance2BetweenPoints( currentImagePoints->GetPoint( i ),
                                                          targetImagePoints->GetPoint( i ) ) );
      }
      else
      {
        sumDist += searchLength;
      }
    }
    avgDist = sumDist / numPoints;
    //avgDist = computeAverageDistanceBetweenTwoPointSets( currentImagePoints, targetImagePoints );
    
    if (verbose)
    {
      std::cout << (fitMode == POSE_ONLY ? "[P] " : (fitMode == SHAPE_ONLY ? "[S] " : "[P+S] "));
      std::cout << "Iteration: " << m << " -----" << std::endl;
      std::cout << "   search range [" << -searchLength/2.0 << " " << searchLength/2.0 << "]" << std::endl;
      std::cout << "   average distance to target image points: " << avgDist << std::endl;
      std::cout << "   max magnitude amaong all sample directions: " << maxMaxEval << std::endl;
      std::cout << "   number of points where target points could not be found: " << numNotFound << std::endl;
    }
    else
    {
      std::cout << "." << std::flush;
    }

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

    // store target points in image space to save them for correspondence purpose (DT mode)
    vnl_vector< double > targetImage( numPoints * 3 );
    vtkPointsToVNLVector( targetImagePoints, targetImage );
    _shapeModel.setImagePoints( targetImage );
    
    // set the scale & pose to shape model
    _shapeModel.setTransform( modelToImageTransform );

    vtkSmartPointer< vtkPoints > pointsAfter = _shapeModel.getPolyData()->GetPoints();

    // measure differences in image space
    double avgDistDiff, maxDistDiff;
    computeDistanceBetweenTwoPointSets( pointsBefore, pointsAfter, avgDistDiff, maxDistDiff );

    if (verbose)
    {
      std::cout << "   maximum distance from previous iteration: " << maxDistDiff << ", average: " << avgDistDiff << std::endl;
    }

    if (maxDistDiff < threshold)
    {
      if (fitMode == POSE_ONLY)
      {
        // pose converged. switching to shape fitting
        fitMode = POSE_AND_SHAPE;
      }
      else
      {
        std::cout << "Stopping at iteration " << m << "." << std::endl;
        break;
      }
    }
  } // for m (repeat shape & pose estimation)
  // *******************************************************************
  std::cout << std::endl;
  
  // save the last average distance to a temporary file
  std::string avg_dist_file = "avg-dist.txt";
  std::cout << "Saving the last average distance: " << avgDist << " to file: " << avg_dist_file << std::endl;
  std::ofstream ofs(avg_dist_file.c_str(), std::ofstream::app);
  ofs << inputFileName << "," << avgDist << "," << numNotFound << "," << avgDist * numNotFound << std::endl;
  ofs.close();
}
