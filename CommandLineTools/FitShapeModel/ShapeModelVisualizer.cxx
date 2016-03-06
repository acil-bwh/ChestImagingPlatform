#include "ShapeModelVisualizer.h"
#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include <vnl/vnl_inverse.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include <vtkXMLPolyDataWriter.h>
#include <climits>
#include <sstream>

ShapeModelVisualizer::ShapeModelVisualizer( const ShapeModel& shapeModel,
                                            MeshType::Pointer mesh,
                                            ImageType::Pointer image,
                                            const std::string& outputName,
                                            const std::string& outputGeomName )
: _shapeModel( shapeModel ),
  _mesh( mesh ),
  _image( image ),
  _outputName( outputName ),
  _outputGeomName( outputGeomName )
{
}

ShapeModelVisualizer::~ShapeModelVisualizer()
{
}

void
ShapeModelVisualizer::updateMeshPoints( vtkSmartPointer< vtkPoints > vpoints )
{
  // update itk mesh points based on the current shape model
  PointsContainerPointer points = _mesh->GetPoints();

  double vp[3];
  unsigned int i = 0;
  for (PointsIterator pit = points->Begin(); pit != points->End(); pit++)
  {
    vpoints->GetPoint( i++, vp );
    PointType& p = pit.Value();
    p[0] = vp[0]; p[1] = vp[1]; p[2] = vp[2];
  }
}

void
ShapeModelVisualizer::createGradientMagnitudeImage( double sigma )
{
  GradientMagnitudeRecursiveGaussianImageFilterType::Pointer gradientMagnitudeFilter = GradientMagnitudeRecursiveGaussianImageFilterType::New();
  gradientMagnitudeFilter->SetInput( _image );
  gradientMagnitudeFilter->SetSigma( sigma );
  try
  {
    std::cout << "Running gradient magnitude filter..." << std::endl;
    gradientMagnitudeFilter->Update();
    std::cout << "Done." << std::endl;
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
  _gradientMagnitudeImage = gradientMagnitudeFilter->GetOutput();
}

void
ShapeModelVisualizer::update( double sigma, int iteration /*=0*/)
{
  updateMeshPoints( _shapeModel.getPolyData()->GetPoints() );

  std::cout << "Creating binary image from mesh..." << std::endl;
  TriangleMeshToBinaryImageFilterType::Pointer meshFilter = TriangleMeshToBinaryImageFilterType::New();
  meshFilter->SetTolerance( 1.0 );
  meshFilter->SetSpacing( _image->GetSpacing() );
  meshFilter->SetOrigin( _image->GetOrigin() );
  meshFilter->SetSize( _image->GetLargestPossibleRegion().GetSize() );
  meshFilter->SetIndex( _image->GetLargestPossibleRegion().GetIndex() );
  meshFilter->SetDirection( _image->GetDirection() );
  meshFilter->SetInput( _mesh );

  try
  {
    meshFilter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    throw std::runtime_error( e.what() );
  }

  ImageType::Pointer meshBinaryImage = meshFilter->GetOutput();
  IteratorType iit( meshBinaryImage, meshBinaryImage->GetLargestPossibleRegion() );
  ConstIteratorType it( _image, _image->GetLargestPossibleRegion() );

  PixelType maxValue = SHRT_MIN;
  PixelType minValue = SHRT_MAX;

  // find scalar value range of the image
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const PixelType& value = it.Get();
    if (value < minValue)
    {
      minValue = value;
    }
    if (value > maxValue)
    {
      maxValue = value;
    }
  }

  // Overlay on top of the the input image
  bool useInputImage = true;

  if (useInputImage)
  {
    for (it.GoToBegin(), iit.GoToBegin(); !iit.IsAtEnd(); ++it, ++iit)
    {
      iit.Value() = it.Value() + (iit.Value() * 0.2 * maxValue); // match the highest intensity of the original image
    }

    // Write the output to file (or new volume in memory)
    std::cout << "Writing output..." << std::endl;
    ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetFileName( _outputName.c_str() );
    writer->UseCompressionOff();
    writer->SetInput( meshBinaryImage );

    try
    {
      writer->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }
  }

  // this flag is mostly for debugging purpose and it takes some time
  // for a larger size of input images
  bool useGradientMagnitudeImage = false;

  if (useGradientMagnitudeImage) // use Gaussian gradient magnitude image for overlay output
  {
    createGradientMagnitudeImage( sigma );

    /*
    IteratorType git( _gradientMagnitudeImage, _gradientMagnitudeImage->GetLargestPossibleRegion() );
    for (git.GoToBegin(), iit.GoToBegin(); !iit.IsAtEnd(); ++git, ++iit)
    {
      git.Value() = git.Value() + (iit.Value() * 200); // match the highest intensity of the original image
    }
    */

    // Write the output to file (or new volume in memory)
    std::cout << "Writing output with gradient magnitude image..." << std::endl;
    ImageWriterType::Pointer writer = ImageWriterType::New();
    std::string magOutputName = _outputName + "-" + int2str(iteration) + ".nrrd";
    writer->SetFileName( magOutputName.c_str() );
    writer->UseCompressionOff();
    writer->SetInput( _gradientMagnitudeImage );

    try
    {
      writer->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }
  }

  // Write output geometry
  if (!_outputGeomName.empty())
  {
    std::cout << "Writing mesh..." << std::endl;

    vtkSmartPointer< vtkMatrix4x4 > matrix = vtkSmartPointer< vtkMatrix4x4 >::New();
    _shapeModel.getTransform()->GetInverse( matrix );

    vtkSmartPointer< vtkTransform > imageToModelTransform = vtkSmartPointer< vtkTransform >::New();
    imageToModelTransform->SetMatrix( matrix );

    vtkSmartPointer< vtkTransformPolyDataFilter > imageToModelTransformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
    imageToModelTransformFilter->SetInputData( _shapeModel.getPolyData() ); // original mean data (model coordinate system)
    imageToModelTransformFilter->SetTransform( imageToModelTransform ); // model coord -> image coord in the Sun's paper
    imageToModelTransformFilter->Update();
    vtkSmartPointer< vtkPolyData > polydata = imageToModelTransformFilter->GetOutput();

    vtkSmartPointer< vtkXMLPolyDataWriter > meshWriter = vtkSmartPointer< vtkXMLPolyDataWriter >::New();
    meshWriter->SetFileName( _outputGeomName.c_str() );
    meshWriter->SetInputData( polydata );
    meshWriter->Write(); // write original mesh

    // save OBJ
    writeVTKPointsToOBJ( polydata->GetPoints(), iteration );

  }
}

void
ShapeModelVisualizer::writeVTKPointsToOBJ( vtkSmartPointer< vtkPoints > points, int iteration )
{
  std::string outputGeomName = _outputGeomName + "-" + int2str(iteration) + ".obj";

  updateMeshPoints( points );

  MeshWriterType::Pointer meshWriter = MeshWriterType::New();
  meshWriter->SetFileName(outputGeomName.c_str());
  meshWriter->SetInput( _mesh );
  try
  {
    meshWriter->Update();
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
}
