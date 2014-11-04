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
#include "cipNewtonOptimizer.h"
#include "cipParticleToThinPlateSplineSurfaceMetric.h"
#include "cipExceptionObject.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRegionOfInterestImageFilter.h"
#include "vtkGraphToPolyData.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderWindow.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkSphereSource.h"
#include "vtkGlyph3D.h"
#include "vtkGraphLayout.h"
#include "vtkGraphLayoutView.h"
#include "vtkProperty.h"
#include "vtkSimple2DLayoutStrategy.h"
#include "vtkGlyphSource2D.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"

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
  outputCT = resizeFilter->GetOutput();

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
    layout->SetInputData( graph );
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

void cip::ViewGraphAsPolyData(vtkSmartPointer< vtkMutableUndirectedGraph > graph)
{
  vtkSmartPointer<vtkGraphToPolyData> graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
    graphToPolyData->SetInputData( graph );
    graphToPolyData->Update();

  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground( 1, 1, 1 );

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(graphToPolyData->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(0, 0, 0);

  renderer->AddActor(actor);

  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

  vtkSmartPointer<vtkInteractorStyleTrackballCamera> trackball = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    renderWindowInteractor->SetInteractorStyle(trackball);

    // Set up the nodes to be rendered
  vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(0.2);
    sphereSource->SetCenter(0, 0, 0);

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

  vtkSmartPointer<vtkPolyData> nodesPoly = vtkSmartPointer<vtkPolyData>::New();
    nodesPoly->SetPoints(graph->GetPoints());

  vtkSmartPointer<vtkGlyph3D> nodesGlyph = vtkSmartPointer<vtkGlyph3D>::New();
    nodesGlyph->SetInputData(nodesPoly);
    nodesGlyph->SetSourceData(sphereSource->GetOutput());
    nodesGlyph->Update();

  vtkSmartPointer<vtkPolyDataMapper> nodesMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    nodesMapper->SetInputData(nodesGlyph->GetOutput());

  vtkSmartPointer<vtkActor> nodesActor = vtkSmartPointer<vtkActor>::New();
    nodesActor->SetMapper(nodesMapper);
    nodesActor->GetProperty()->SetColor(1, 0, 0);

  renderer->AddActor(nodesActor);

  renderWindow->Render();
  renderWindowInteractor->Start();
}

void cip::DilateLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type,
			 unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ)
{
  ChestConventions conventions;

  unsigned short labelMapValue = conventions.GetValueFromChestRegionAndType(region, type);

  typedef itk::BinaryBallStructuringElement<unsigned short, 3>                            ElementType;
  typedef itk::BinaryDilateImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType> DilateType;
  typedef itk::ImageRegionIterator<cip::LabelMapType>                                     IteratorType;
  typedef itk::RegionOfInterestImageFilter<cip::LabelMapType, cip::LabelMapType>          ROIType;

  // We will want to isolate the bounding box containing the chest region - chest type combination
  // of interest. We will only want to perform the closing operation over that region for the
  // sake of speed.
  cip::LabelMapType::RegionType roiPadded =
    cip::GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(labelMap, region, type, kernelRadiusX, kernelRadiusY, kernelRadiusZ);

  if ( roiPadded.GetSize()[0] == 0 && roiPadded.GetSize()[1] == 0 && roiPadded.GetSize()[2] == 0 )
    {
      return;
    }

  ROIType::Pointer roiExtractor = ROIType::New();
    roiExtractor->SetRegionOfInterest(roiPadded);
    roiExtractor->SetInput(labelMap);

  unsigned long neighborhood[3];
    neighborhood[0] = kernelRadiusX;
    neighborhood[1] = kernelRadiusY;
    neighborhood[2] = kernelRadiusZ;

  ElementType structuringElement;
    structuringElement.SetRadius(neighborhood);
    structuringElement.CreateStructuringElement();

  DilateType::Pointer dilater = DilateType::New();
    dilater->SetInput(roiExtractor->GetOutput());
    dilater->SetKernel(structuringElement);
    dilater->SetDilateValue(labelMapValue);
  try
    {
    dilater->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught dilating label map:";
    std::cerr << excp << std::endl;
    }

  IteratorType dIt(dilater->GetOutput(), dilater->GetOutput()->GetBufferedRegion());
  IteratorType lIt(labelMap, roiPadded);

  dIt.GoToBegin();
  lIt.GoToBegin();
  while (!lIt.IsAtEnd())
    {
    lIt.Set(dIt.Get());

    ++dIt;
    ++lIt;
    }
}

void cip::ErodeLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type,
			unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ)
{
  ChestConventions conventions;

  unsigned short labelMapValue = conventions.GetValueFromChestRegionAndType(region, type);

  typedef itk::BinaryBallStructuringElement<unsigned short, 3>                           ElementType;
  typedef itk::BinaryErodeImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType> ErodeType;
  typedef itk::ImageRegionIterator<cip::LabelMapType>                                    IteratorType;
  typedef itk::RegionOfInterestImageFilter<cip::LabelMapType, cip::LabelMapType>         ROIType;

  // We will want to isolate the bounding box containing the chest region - chest type combination
  // of interest. We will only want to perform the closing operation over that region for the
  // sake of speed.
  cip::LabelMapType::RegionType roiPadded =
    cip::GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(labelMap, region, type, kernelRadiusX, kernelRadiusY, kernelRadiusZ);

  if ( roiPadded.GetSize()[0] == 0 && roiPadded.GetSize()[1] == 0 && roiPadded.GetSize()[2] == 0 )
    {
      return;
    }

  ROIType::Pointer roiExtractor = ROIType::New();
    roiExtractor->SetRegionOfInterest(roiPadded);
    roiExtractor->SetInput(labelMap);

  unsigned long neighborhood[3];
    neighborhood[0] = kernelRadiusX;
    neighborhood[1] = kernelRadiusY;
    neighborhood[2] = kernelRadiusZ;

  ElementType structuringElement;
    structuringElement.SetRadius(neighborhood);
    structuringElement.CreateStructuringElement();

  ErodeType::Pointer eroder = ErodeType::New();
    eroder->SetInput(roiExtractor->GetOutput());
    eroder->SetKernel(structuringElement);
    eroder->SetErodeValue(labelMapValue);
  try
    {
    eroder->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught dilating label map:";
    std::cerr << excp << std::endl;
    }

  IteratorType eIt(eroder->GetOutput(), eroder->GetOutput()->GetBufferedRegion());
  IteratorType lIt(labelMap, roiPadded);

  eIt.GoToBegin();
  lIt.GoToBegin();
  while (!lIt.IsAtEnd())
    {
    lIt.Set(eIt.Get());

    ++eIt;
    ++lIt;
    }
}

void cip::CloseLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type,
		     unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ)
{
  ChestConventions conventions;

  unsigned short labelMapValue = conventions.GetValueFromChestRegionAndType(region, type);

  typedef itk::BinaryBallStructuringElement<unsigned short, 3>                            ElementType;
  typedef itk::BinaryDilateImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType> DilateType;
  typedef itk::BinaryErodeImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType>  ErodeType;
  typedef itk::ImageRegionIterator<cip::LabelMapType>                                     IteratorType;
  typedef itk::RegionOfInterestImageFilter<cip::LabelMapType, cip::LabelMapType>          ROIType;

  // We will want to isolate the bounding box containing the chest region - chest type combination
  // of interest. We will only want to perform the closing operation over that region for the
  // sake of speed.
  cip::LabelMapType::RegionType roiPadded =
    cip::GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(labelMap, region, type, kernelRadiusX*3, kernelRadiusY*3, kernelRadiusZ*3);

  if ( roiPadded.GetSize()[0] == 0 && roiPadded.GetSize()[1] == 0 && roiPadded.GetSize()[2] == 0 )
    {
      return;
    }

  ROIType::Pointer roiExtractor = ROIType::New();
    roiExtractor->SetRegionOfInterest(roiPadded);
    roiExtractor->SetInput(labelMap);

  // Set up the kernel
  unsigned long neighborhood[3];
    neighborhood[0] = kernelRadiusX;
    neighborhood[1] = kernelRadiusY;
    neighborhood[2] = kernelRadiusZ;

  ElementType structuringElement;
    structuringElement.SetRadius(neighborhood);
    structuringElement.CreateStructuringElement();

  // Perform dilation on the extracted region
  DilateType::Pointer dilater = DilateType::New();
    dilater->SetInput(roiExtractor->GetOutput());
    dilater->SetKernel(structuringElement);
    dilater->SetDilateValue(labelMapValue);
  try
    {
    dilater->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught dilating label map:";
    std::cerr << excp << std::endl;
    }

  ErodeType::Pointer eroder = ErodeType::New();
    eroder->SetInput(dilater->GetOutput());
    eroder->SetKernel(structuringElement);
    eroder->SetErodeValue(labelMapValue);
  try
    {
    eroder->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught eroding label map:";
    std::cerr << excp << std::endl;
    }

  // Now replace the label map values with those from the closing operation. Note that
  // the iterator over the erosion output is over the entire region but that we only
  // iterate over the label map in the area defined above -- the area that was
  // extracted from the original label map to operate on.
  IteratorType eIt(eroder->GetOutput(), eroder->GetOutput()->GetBufferedRegion());
  IteratorType lIt(labelMap, roiPadded);

  eIt.GoToBegin();
  lIt.GoToBegin();
  while (!lIt.IsAtEnd())
    {
    lIt.Set(eIt.Get());

    ++eIt;
    ++lIt;
    }
}

void cip::OpenLabelMap(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type,
		       unsigned int kernelRadiusX, unsigned int kernelRadiusY, unsigned int kernelRadiusZ)
{
  ChestConventions conventions;

  unsigned short labelMapValue = conventions.GetValueFromChestRegionAndType(region, type);

  typedef itk::BinaryBallStructuringElement<unsigned short, 3>                            ElementType;
  typedef itk::BinaryErodeImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType>  ErodeType;
  typedef itk::BinaryDilateImageFilter<cip::LabelMapType, cip::LabelMapType, ElementType> DilateType;
  typedef itk::ImageRegionIterator<cip::LabelMapType>                                     IteratorType;
  typedef itk::RegionOfInterestImageFilter<cip::LabelMapType, cip::LabelMapType>          ROIType;

  // We will want to isolate the bounding box containing the chest region - chest type combination
  // of interest. We will only want to perform the closing operation over that region for the
  // sake of speed.
  cip::LabelMapType::RegionType roiPadded =
    cip::GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(labelMap, region, type, kernelRadiusX, kernelRadiusY, kernelRadiusZ);

  if ( roiPadded.GetSize()[0] == 0 && roiPadded.GetSize()[1] == 0 && roiPadded.GetSize()[2] == 0 )
    {
      return;
    }

  ROIType::Pointer roiExtractor = ROIType::New();
    roiExtractor->SetRegionOfInterest(roiPadded);
    roiExtractor->SetInput(labelMap);

  unsigned long neighborhood[3];
    neighborhood[0] = kernelRadiusX;
    neighborhood[1] = kernelRadiusY;
    neighborhood[2] = kernelRadiusZ;

  ElementType structuringElement;
    structuringElement.SetRadius(neighborhood);
    structuringElement.CreateStructuringElement();

  ErodeType::Pointer eroder = ErodeType::New();
    eroder->SetInput(roiExtractor->GetOutput());
    eroder->SetKernel(structuringElement);
    eroder->SetErodeValue(labelMapValue);
  try
    {
    eroder->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught dilating label map:";
    std::cerr << excp << std::endl;
    }

  DilateType::Pointer dilater = DilateType::New();
    dilater->SetInput(eroder->GetOutput());
    dilater->SetKernel(structuringElement);
    dilater->SetDilateValue(labelMapValue);
  try
    {
    dilater->Update();
    }
  catch(itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught dilating label map:";
    std::cerr << excp << std::endl;
    }

  IteratorType dIt(dilater->GetOutput(), dilater->GetOutput()->GetBufferedRegion());
  IteratorType lIt(labelMap, roiPadded);

  dIt.GoToBegin();
  lIt.GoToBegin();
  while (!lIt.IsAtEnd())
    {
    lIt.Set(dIt.Get());

    ++dIt;
    ++lIt;
    }
}

void cip::AssertChestRegionChestTypeArrayExistence( vtkSmartPointer< vtkPolyData > particles )
{
  unsigned int numberParticles         = particles->GetNumberOfPoints();
  unsigned int numberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  bool foundChestRegionArray = false;
  bool foundChestTypeArray   = false;

  for ( unsigned int i=0; i<numberOfPointDataArrays; i++ )
    {
    std::string name( particles->GetPointData()->GetArray(i)->GetName() );

    if ( name.compare( "ChestRegion" ) == 0 )
      {
      foundChestRegionArray = true;
      }
    if ( name.compare( "ChestType" ) == 0 )
      {
      foundChestTypeArray = true;
      }
    }

  if ( !foundChestRegionArray )
    {
    vtkSmartPointer< vtkFloatArray > chestRegionArray = vtkSmartPointer< vtkFloatArray >::New();
      chestRegionArray->SetNumberOfComponents( 1 );
      chestRegionArray->SetName( "ChestRegion" );

    particles->GetPointData()->AddArray( chestRegionArray );
    }
  if ( !foundChestTypeArray )
    {
    vtkSmartPointer< vtkFloatArray > chestTypeArray = vtkSmartPointer< vtkFloatArray >::New();
      chestTypeArray->SetNumberOfComponents( 1 );
      chestTypeArray->SetName( "ChestType" );

    particles->GetPointData()->AddArray( chestTypeArray );
    }

  float cipRegion = static_cast< float >( cip::UNDEFINEDREGION );
  float cipType   = static_cast< float >( cip::UNDEFINEDTYPE );
  if ( !foundChestRegionArray || !foundChestTypeArray )
    {
    for ( unsigned int i=0; i<numberParticles; i++ )
      {
      if ( !foundChestRegionArray )
        {
        particles->GetPointData()->GetArray( "ChestRegion" )->InsertTuple( i, &cipRegion );
        }
      if ( !foundChestTypeArray )
        {
        particles->GetPointData()->GetArray( "ChestType" )->InsertTuple( i, &cipType );
        }
      }
    }
}

cip::LabelMapType::RegionType cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(cip::LabelMapType::Pointer labelMap,
										    unsigned char cipRegion, unsigned char cipType)
{
  ChestConventions conventions;

  unsigned short value = conventions.GetValueFromChestRegionAndType(cipRegion, cipType);

  unsigned int xMin = labelMap->GetBufferedRegion().GetSize()[0];
  unsigned int xMax = 0;
  unsigned int yMin = labelMap->GetBufferedRegion().GetSize()[1];
  unsigned int yMax = 0;
  unsigned int zMin = labelMap->GetBufferedRegion().GetSize()[2];
  unsigned int zMax = 0;

  typedef itk::ImageRegionIteratorWithIndex<cip::LabelMapType> IteratorType;

  IteratorType it(labelMap, labelMap->GetBufferedRegion());

  it.GoToBegin();
  while (!it.IsAtEnd())
    {
    if (it.Get() > 0)
      {
	unsigned char currentRegion = conventions.GetChestRegionFromValue( it.Get() );
	unsigned char currentType   = conventions.GetChestTypeFromValue( it.Get() );

      // By default 'value' is zero, indicating that we want the bounding box over
      // the entire foreground region. So if either the foreground value is equal to
      // the requested region-type pair, or if we want to consider the foreground as
      // a whole, we will update the bounding box info.
      if ( (currentType == cipType && conventions.CheckSubordinateSuperiorChestRegionRelationship( currentRegion, cipRegion)) || value == 0 )
	{
	if (it.GetIndex()[0] < xMin)
	  {
	  xMin = it.GetIndex()[0];
	  }
	if (it.GetIndex()[0] > xMax)
	  {
	  xMax = it.GetIndex()[0];
	  }
	if (it.GetIndex()[1] < yMin)
	  {
	  yMin = it.GetIndex()[1];
	  }
	if (it.GetIndex()[1] > yMax)
	  {
	  yMax = it.GetIndex()[1];
	  }
	if (it.GetIndex()[2] < zMin)
	  {
	  zMin = it.GetIndex()[2];
	  }
	if (it.GetIndex()[2] > zMax)
	  {
	  zMax = it.GetIndex()[2];
	  }
	}
      }

    ++it;
    }

  cip::LabelMapType::SizeType size;
    size[0] = xMax - xMin + 1;
    size[1] = yMax - yMin + 1;
    size[2] = zMax - zMin + 1;

  cip::LabelMapType::IndexType start;
    start[0] = xMin;
    start[1] = yMin;
    start[2] = zMin;

  // Check that Bounding Box and Size have been set. If not, set the region
  // size to have no extent
  if ( xMin > xMax || yMin > yMax || zMin > zMax )
    {
      size[0] = 0;
      size[1] = 0;
      size[2] = 0;

      start[0] = 0;
      start[1] = 0;
      start[2] = 0;
    }

  cip::LabelMapType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

  return region;
}

cip::LabelMapType::RegionType cip::GetLabelMapChestRegionChestTypePaddedBoundingBoxRegion(cip::LabelMapType::Pointer labelMap, unsigned char region, unsigned char type,
											  unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ)
{
  // Get the bounding box region prior to padding
  cip::LabelMapType::RegionType roi =
    cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(labelMap, region, type);

  if ( roi.GetSize()[0] == 0 && roi.GetSize()[1] == 0 && roi.GetSize()[2] == 0 )
    {
      return roi;
    }

  // Now construct the padded bounding box
  cip::LabelMapType::IndexType roiPaddedStart;

  int tmpStart[3];
    tmpStart[0] = int(roi.GetIndex()[0]) - int(radiusX);
    tmpStart[1] = int(roi.GetIndex()[1]) - int(radiusY);
    tmpStart[2] = int(roi.GetIndex()[2]) - int(radiusZ);

  if (tmpStart[0] < 0)
    {
    roiPaddedStart[0] = 0;
    }
  else
    {
    roiPaddedStart[0] = tmpStart[0];
    }
  if (tmpStart[1] < 0)
    {
    roiPaddedStart[1] = 0;
    }
  else
    {
    roiPaddedStart[1] = tmpStart[1];
    }
  if (tmpStart[2] < 0)
    {
    roiPaddedStart[2] = 0;
    }
  else
    {
    roiPaddedStart[2] = tmpStart[2];
    }

  // Get the size of the padded region. Note that we need to check that the region does not
  // extend past the bounds of the image.
  cip::LabelMapType::SizeType roiPaddedSize;

  int tmpSize[3];
    tmpSize[0] = roi.GetSize()[0] + 2*radiusX;
    tmpSize[1] = roi.GetSize()[1] + 2*radiusY;
    tmpSize[2] = roi.GetSize()[2] + 2*radiusZ;

  if (roiPaddedStart[0] + tmpSize[0] > labelMap->GetBufferedRegion().GetSize()[0])
    {
    roiPaddedSize[0] = labelMap->GetBufferedRegion().GetSize()[0] - roiPaddedStart[0];
    }
  else
    {
    roiPaddedSize[0] = tmpSize[0];
    }
  if (roiPaddedStart[1] + tmpSize[1] > labelMap->GetBufferedRegion().GetSize()[1])
    {
    roiPaddedSize[1] = labelMap->GetBufferedRegion().GetSize()[1] - roiPaddedStart[1];
    }
  else
    {
    roiPaddedSize[1] = tmpSize[1];
    }
  if (roiPaddedStart[2] + tmpSize[2] > labelMap->GetBufferedRegion().GetSize()[2])
    {
    roiPaddedSize[2] = labelMap->GetBufferedRegion().GetSize()[2] - roiPaddedStart[2];
    }
  else
    {
    roiPaddedSize[2] = tmpSize[2];
    }

  // Now we can set up the region extractor
  cip::LabelMapType::RegionType roiPadded;
    roiPadded.SetSize(roiPaddedSize);
    roiPadded.SetIndex(roiPaddedStart);

  return roiPadded;
}

void cip::GraftPointDataArrays( vtkSmartPointer< vtkPolyData > fromPoly, vtkSmartPointer< vtkPolyData > toPoly )
{
  assert( fromPoly->GetNumberOfPoints() == toPoly->GetNumberOfPoints() );

  unsigned int numFromArrays = fromPoly->GetPointData()->GetNumberOfArrays();
  unsigned int numToArrays   = toPoly->GetPointData()->GetNumberOfArrays();

  bool alreadyPresent;
  for ( unsigned int i=0; i<numFromArrays; i++ )
    {
      alreadyPresent = false;
      std::string fromArrayName = fromPoly->GetPointData()->GetArray(i)->GetName();
      for ( unsigned int j=0; j<numToArrays; j++ )
	{
	  std::string toArrayName = toPoly->GetPointData()->GetArray(j)->GetName();
	  if ( fromArrayName.compare( toArrayName ) == 0 )
	    {
	      alreadyPresent = true;
	      break;
	    }
	}

      if ( !alreadyPresent )
	{
	  toPoly->GetPointData()->AddArray( fromPoly->GetPointData()->GetArray(i) );
	}
    }
}

double cip::GetDistanceToThinPlateSplineSurface( cipThinPlateSplineSurface* tps, double* point )
{
  cipNewtonOptimizer< 2 >::PointType* domainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  (*domainParams)[0] = point[0];
  (*domainParams)[1] = point[1];

  cipParticleToThinPlateSplineSurfaceMetric* tpsMetric = new cipParticleToThinPlateSplineSurfaceMetric();
    tpsMetric->SetThinPlateSplineSurface( tps );
    tpsMetric->SetParticle( point );

  cipNewtonOptimizer< 2 >* optimizer = new cipNewtonOptimizer< 2 >();
    optimizer->SetMetric( tpsMetric );
    optimizer->SetInitialParameters( domainParams );
    optimizer->Update();

  double distance = vcl_sqrt( optimizer->GetOptimalValue() );

  delete domainParams;
  delete tpsMetric;
  delete optimizer;

  return distance ;
}
