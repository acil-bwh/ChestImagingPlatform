/**
*
*  $Date: $
*  $Revision: $
*  $Author: $
*
*  Class cipHelper
*	
*/

#include "cipHelper.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "vtkGraphToPolyData.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderWindow.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkSphereSource.h"
#include "vtkPolyData.h"
#include "vtkGlyph3D.h"
#include "vtkGraphLayout.h"
#include "vtkGraphLayoutView.h"
#include "vtkProperty.h"
#include "vtkSimple2DLayoutStrategy.h"
#include "vtkGlyphSource2D.h"

//
// Code modified from //http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/Upsampling
//
cip::LabelMapType::Pointer cip::DownsampleLabelMap(short samplingAmount, cip::LabelMapType::Pointer inputLabelMap)
{
  cip::LabelMapType::Pointer outputLabelMap;

  typedef itk::IdentityTransform<double, 3>                                        TransformType;
  typedef itk::NearestNeighborInterpolateImageFunction<cip::LabelMapType, double>  InterpolatorType;
  typedef itk::ResampleImageFilter<cip::LabelMapType, cip::LabelMapType>           ResampleType;

  // Instantiate the transform, the nearest-neighbour interpolator and the resampler
  TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetIdentity();

  InterpolatorType::Pointer imageInterpolator = InterpolatorType::New();

  // Compute and set the output spacing from the input spacing and samplingAmount  
  const cip::LabelMapType::RegionType& inputRegion = inputLabelMap->GetLargestPossibleRegion();
  const cip::LabelMapType::SizeType& inputSize = inputRegion.GetSize();

  unsigned int originalWidth  = inputSize[0];
  unsigned int originalLength = inputSize[1];
  unsigned int originalHeight = inputSize[2];

  unsigned int newWidth  = (unsigned int)(double(originalWidth)/double(samplingAmount));
  unsigned int newLength = (unsigned int)(double(originalLength)/double(samplingAmount));
  unsigned int newHeight = (unsigned int)(double(originalHeight)/double(samplingAmount));

  const cip::LabelMapType::SpacingType& inputSpacing = inputLabelMap->GetSpacing();

  double outputSpacing[3];
    outputSpacing[0] = inputSpacing[0]*(double(originalWidth)/double(newWidth));
    outputSpacing[1] = inputSpacing[1]*(double(originalLength)/double(newLength));
    outputSpacing[2] = inputSpacing[2]*(double(originalHeight)/double(newHeight));
  
  // Set the resampler with the calculated parameters and resample  
  itk::Size< 3 > outputSize = { {newWidth, newLength, newHeight} };

  ResampleType::Pointer resizeFilter = ResampleType::New();
    resizeFilter->SetTransform( idTransform );
    resizeFilter->SetInterpolator( imageInterpolator );
    resizeFilter->SetOutputOrigin( inputLabelMap->GetOrigin() );
    resizeFilter->SetOutputSpacing( outputSpacing );
    resizeFilter->SetSize( outputSize );
    resizeFilter->SetInput( inputLabelMap );
    resizeFilter->Update();

  // Save the resampled output to the output image and return
  outputLabelMap = resizeFilter->GetOutput();
  
  return outputLabelMap;
}

cip::LabelMapType::Pointer cip::UpsampleLabelMap(short samplingAmount, cip::LabelMapType::Pointer inputLabelMap)
{
  cip::LabelMapType::Pointer outputLabelMap;

  typedef itk::IdentityTransform<double, 3>                                        TransformType;
  typedef itk::NearestNeighborInterpolateImageFunction<cip::LabelMapType, double>  InterpolatorType;
  typedef itk::ResampleImageFilter<cip::LabelMapType, cip::LabelMapType>           ResampleType;

  // Instantiate the transform, the b-spline interpolator and the resampler
  TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetIdentity();

  InterpolatorType::Pointer imageInterpolator = InterpolatorType::New();

  // Compute and set the output spacing from the input spacing and samplingAmount 
  const LabelMapType::RegionType& inputRegion = inputLabelMap->GetLargestPossibleRegion();
  const LabelMapType::SizeType& inputSize = inputRegion.GetSize();

  unsigned int originalWidth = inputSize[0];
  unsigned int originalLength = inputSize[1];
  unsigned int originalHeight = inputSize[2];

  unsigned int newWidth  = (unsigned int)(double(originalWidth)*double(samplingAmount));
  unsigned int newLength = (unsigned int)(double(originalLength)*double(samplingAmount));
  unsigned int newHeight = (unsigned int)(double(originalHeight)*double(samplingAmount));

  const cip::LabelMapType::SpacingType& inputSpacing = inputLabelMap->GetSpacing();

  double outputSpacing[3];
    outputSpacing[0] = inputSpacing[0]*(double(originalWidth)/double(newWidth));
    outputSpacing[1] = inputSpacing[1]*(double(originalLength)/double(newLength));
    outputSpacing[2] = inputSpacing[2]*(double(originalHeight)/double(newHeight));
    
  // Set the resampler with the calculated parameters and resample
  itk::Size< 3 > outputSize = { {newWidth, newLength, newHeight} };

  ResampleType::Pointer resizeFilter = ResampleType::New();
    resizeFilter->SetTransform( idTransform );
    resizeFilter->SetInterpolator( imageInterpolator );
    resizeFilter->SetOutputOrigin( inputLabelMap->GetOrigin() );
    resizeFilter->SetOutputSpacing( outputSpacing );
    resizeFilter->SetSize( outputSize );
    resizeFilter->SetInput( inputLabelMap );
    resizeFilter->Update();

  // Save the resampled output to the output image and return
  outputLabelMap= resizeFilter->GetOutput();

  return outputLabelMap;
}

cip::CTType::Pointer cip::UpsampleCT(short samplingAmount, cip::CTType::Pointer inputCT)
{
  cip::CTType::Pointer outputCT;

  typedef itk::IdentityTransform<double, 3>                          TransformType;
  typedef itk::LinearInterpolateImageFunction<cip::CTType, double>   InterpolatorType;
  typedef itk::ResampleImageFilter<cip::CTType, cip::CTType>	     ResampleType;

  // Instantiate the transform, the linear interpolator and the resampler
  TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetIdentity();

  InterpolatorType::Pointer imageInterpolator = InterpolatorType::New();

  // Compute and set the output spacing from the input spacing and samplingAmount 
  const cip::CTType::RegionType& inputRegion = inputCT->GetLargestPossibleRegion();
  const cip::CTType::SizeType& inputSize = inputRegion.GetSize();

  unsigned int originalWidth  = inputSize[0];
  unsigned int originalLength = inputSize[1];
  unsigned int originalHeight = inputSize[2];

  unsigned int newWidth  = (unsigned int)(double(originalWidth)*double(samplingAmount)); 
  unsigned int newLength = (unsigned int)(double(originalLength)*double(samplingAmount));
  unsigned int newHeight = (unsigned int)(double(originalHeight)*double(samplingAmount));

  const cip::CTType::SpacingType& inputSpacing = inputCT->GetSpacing();

  double outputSpacing[3];
  outputSpacing[0] = inputSpacing[0]*(double(originalWidth)/double(newWidth));
  outputSpacing[1] = inputSpacing[1]*(double(originalLength)/double(newLength)); 
  outputSpacing[2] = inputSpacing[2]*(double(originalHeight)/double(newHeight));

  // Set the resampler with the calculated parameters and resample
  itk::Size< 3 > outputSize = { {newWidth, newLength, newHeight} };

  ResampleType::Pointer resizeFilter = ResampleType::New();
    resizeFilter->SetTransform( idTransform );
    resizeFilter->SetInterpolator( imageInterpolator );
    resizeFilter->SetOutputOrigin( inputCT->GetOrigin() );
    resizeFilter->SetOutputSpacing( outputSpacing );
    resizeFilter->SetSize( outputSize );
    resizeFilter->SetInput( inputCT );
    resizeFilter->Update();

  // Save the resampled output to the output image and return
  outputCT= resizeFilter->GetOutput();

  return outputCT;
}

cip::CTType::Pointer cip::DownsampleCT(short samplingAmount, cip::CTType::Pointer inputCT)
{
  cip::CTType::Pointer outputCT;

  typedef itk::IdentityTransform<double, 3>                         TransformType;
  typedef itk::LinearInterpolateImageFunction<cip::CTType, double>  InterpolatorType;
  typedef itk::ResampleImageFilter<cip::CTType, cip::CTType>	    ResampleType;

  // Instantiate the transform, the linear interpolator and the resampler
  TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetIdentity();

  InterpolatorType::Pointer imageInterpolator = InterpolatorType::New();

  // Compute and set the output spacing from the input spacing and samplingAmount 
  const cip::CTType::RegionType& inputRegion = inputCT->GetLargestPossibleRegion();
  const cip::CTType::SizeType& inputSize = inputRegion.GetSize();

  unsigned int originalWidth = inputSize[0];
  unsigned int originalLength = inputSize[1];
  unsigned int originalHeight = inputSize[2];

  unsigned int newWidth  = (unsigned int)(double(originalWidth)/double(samplingAmount)); 
  unsigned int newLength = (unsigned int)(double(originalLength)/double(samplingAmount));
  unsigned int newHeight = (unsigned int)(double(originalHeight)/double(samplingAmount));

  const cip::CTType::SpacingType& inputSpacing = inputCT->GetSpacing();

  double outputSpacing[3];
    outputSpacing[0] = inputSpacing[0]*(double(originalWidth)/double(newWidth));
    outputSpacing[1] = inputSpacing[1]*(double(originalLength)/double(newLength)); 
    outputSpacing[2] = inputSpacing[2]*(double(originalHeight)/double(newHeight));

  // Set the resampler with the calculated parameters and resample
  itk::Size< 3 > outputSize = { {newWidth, newLength, newHeight} };

  ResampleType::Pointer resizeFilter = ResampleType::New();
    resizeFilter->SetTransform( idTransform );
    resizeFilter->SetInterpolator( imageInterpolator );
    resizeFilter->SetOutputOrigin( inputCT->GetOrigin() );
    resizeFilter->SetOutputSpacing( outputSpacing );
    resizeFilter->SetSize( outputSize );
    resizeFilter->SetInput( inputCT );
    resizeFilter->Update();

    // Save the resampled output to the output image and return
  outputCT= resizeFilter->GetOutput();

  return outputCT;
}


double cip::GetVectorMagnitude(double vector[3])
{
  double magnitude = vcl_sqrt(std::pow(vector[0], 2) + std::pow(vector[1], 2) + std::pow(vector[2], 2));

  return magnitude;
}


double cip::GetAngleBetweenVectors(double vec1[3], double vec2[3], bool returnDegrees)
{
  double vec1Mag = cip::GetVectorMagnitude(vec1);
  double vec2Mag = cip::GetVectorMagnitude(vec2);

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( vcl_abs( arg ) > 1.0 )
    {
      arg = 1.0;
    }
  
  double angle = vcl_acos( arg );

  if ( !returnDegrees )
    {
      return angle;
    }

  double angleInDegrees = (180.0/vnl_math::pi)*angle;

  if ( angleInDegrees > 90.0 )
    {
      angleInDegrees = 180.0 - angleInDegrees;
    }

  return angleInDegrees;
}

void cip::ViewGraph( vtkSmartPointer< vtkMutableDirectedGraph > graph )
{ 
  vtkSmartPointer< vtkSimple2DLayoutStrategy > strategy = vtkSmartPointer< vtkSimple2DLayoutStrategy >::New();

  vtkSmartPointer< vtkGraphLayout > layout =  vtkSmartPointer< vtkGraphLayout >::New();
    layout->SetInput( graph );
    layout->SetLayoutStrategy( strategy );

  vtkSmartPointer< vtkGraphToPolyData > graphToPoly = vtkSmartPointer< vtkGraphToPolyData >::New();
    graphToPoly->SetInputConnection( layout->GetOutputPort() );
    graphToPoly->EdgeGlyphOutputOn();
    graphToPoly->SetEdgeGlyphPosition(0.98);

  vtkSmartPointer< vtkGlyphSource2D > arrowSource = vtkSmartPointer< vtkGlyphSource2D >::New();
    arrowSource->SetGlyphTypeToEdgeArrow();
    arrowSource->SetScale(0.1);
    arrowSource->Update();

  vtkSmartPointer< vtkGlyph3D > arrowGlyph = vtkSmartPointer< vtkGlyph3D >::New();
    arrowGlyph->SetInputConnection( 0, graphToPoly->GetOutputPort(1) );
    arrowGlyph->SetInputConnection( 1, arrowSource->GetOutputPort()) ;

  vtkSmartPointer< vtkPolyDataMapper > arrowMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    arrowMapper->SetInputConnection(arrowGlyph->GetOutputPort());

  vtkSmartPointer< vtkActor > arrowActor =  vtkSmartPointer< vtkActor >::New();
    arrowActor->SetMapper(arrowMapper);

  vtkSmartPointer< vtkGraphLayoutView > graphLayoutView = vtkSmartPointer< vtkGraphLayoutView >::New();
    graphLayoutView->SetLayoutStrategyToPassThrough();
    graphLayoutView->SetEdgeLayoutStrategyToPassThrough(); 
    graphLayoutView->AddRepresentationFromInputConnection( layout->GetOutputPort() );
    graphLayoutView->GetRenderer()->AddActor( arrowActor ); 
    graphLayoutView->ResetCamera();
    graphLayoutView->Render();
    graphLayoutView->GetInteractor()->Start();
}


void cip::ViewGraphAsPolyData( vtkSmartPointer< vtkMutableUndirectedGraph > graph )
{
  vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
    graphToPolyData->SetInput( graph );
    graphToPolyData->Update();

  vtkSmartPointer< vtkRenderer > renderer = vtkSmartPointer< vtkRenderer >::New();
    renderer->SetBackground( 1, 1, 1 ); 

  vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    mapper->SetInputConnection( graphToPolyData->GetOutputPort() );

  vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
    actor->SetMapper( mapper );
    actor->GetProperty()->SetColor( 0, 0, 0 );

  renderer->AddActor( actor );

  vtkSmartPointer< vtkRenderWindow > renderWindow = vtkSmartPointer< vtkRenderWindow >::New();
    renderWindow->AddRenderer( renderer );

  vtkSmartPointer< vtkInteractorStyleTrackballCamera > trackball = vtkSmartPointer< vtkInteractorStyleTrackballCamera >::New();

  vtkSmartPointer< vtkRenderWindowInteractor > renderWindowInteractor = vtkSmartPointer< vtkRenderWindowInteractor >::New();
    renderWindowInteractor->SetRenderWindow( renderWindow );
    renderWindowInteractor->SetInteractorStyle( trackball );
    
    // Set up the nodes to be rendered
  vtkSmartPointer< vtkSphereSource > sphereSource = vtkSmartPointer< vtkSphereSource >::New();
    sphereSource->SetRadius( 0.2 );
    sphereSource->SetCenter( 0, 0, 0 );

    //   vtkSmartPointer< vtkPoints > leafPoints = vtkSmartPointer< vtkPoints >::New();
    //   for ( unsigned int i=0; i<this->SubGraphLeafParticleIDs.size(); i++ )
    //     {
    //     unsigned int leafParticleID  = this->SubGraphLeafParticleIDs[i];
    
    //     double leafPoint[3];
    //     leafPoint[0] = this->InternalInputPolyData->GetPoint( leafParticleID )[0];
    //     leafPoint[1] = this->InternalInputPolyData->GetPoint( leafParticleID )[1];
    //     leafPoint[2] = this->InternalInputPolyData->GetPoint( leafParticleID )[2];
    
    //     leafPoints->InsertNextPoint( leafPoint[0], leafPoint[1], leafPoint[2] );
    //     }

  vtkSmartPointer< vtkPolyData > nodesPoly = vtkSmartPointer< vtkPolyData >::New();
    nodesPoly->SetPoints( graph->GetPoints() );

  vtkSmartPointer< vtkGlyph3D > nodesGlyph = vtkSmartPointer< vtkGlyph3D >::New();
    nodesGlyph->SetInput( nodesPoly );
    nodesGlyph->SetSource( sphereSource->GetOutput() );
    nodesGlyph->Update();
    
  vtkSmartPointer< vtkPolyDataMapper > nodesMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    nodesMapper->SetInput( nodesGlyph->GetOutput() );

  vtkSmartPointer< vtkActor > nodesActor = vtkSmartPointer< vtkActor >::New();
    nodesActor->SetMapper( nodesMapper );
    nodesActor->GetProperty()->SetColor( 1, 0, 0 );

  renderer->AddActor( nodesActor );

  renderWindow->Render();
  renderWindowInteractor->Start();
}
