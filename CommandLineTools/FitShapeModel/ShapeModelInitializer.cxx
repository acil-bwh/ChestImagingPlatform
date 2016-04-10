#include "ShapeModelInitializer.h"
#include "ShapeModel.h"
#include "ShapeModelImage.h"
#include <vtkCenterOfMass.h>

ShapeModelInitializer::ShapeModelInitializer( ShapeModel& shapeModel,
                                              ShapeModelImage& image )
: _shapeModel( shapeModel ),
  _image( image )
{
}

ShapeModelInitializer::~ShapeModelInitializer()
{
}

void 
ShapeModelInitializer::run( const std::string& transformFileName )
{
  TransformFileReaderType::Pointer transformReader = TransformFileReaderType::New();
  transformReader->SetFileName( transformFileName );
  try
  {
    transformReader->Update();
  }
  catch ( itk::ExceptionObject &e )
  {
    throw std::runtime_error( e.what() );
  }
  
  TransformFileReaderType::TransformListType::const_iterator it;
  it = transformReader->GetTransformList()->begin();
  
  TransformType::Pointer itkTransform = static_cast< TransformType* >( (*it).GetPointer() );
  const TransformType::ParametersType parameters = itkTransform->GetParameters();
  
  // convert itk transform parameters to vtk homogeneous 4x4 matrix
  vtkSmartPointer< vtkMatrix4x4 > matrix = vtkSmartPointer< vtkMatrix4x4 >::New();

  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      matrix->SetElement( i, j, parameters[i * Dimension + j] );
    }
    matrix->SetElement( i, Dimension, parameters[i + Dimension * Dimension] );
  }
  
  //std::cout << "Initial transform matrix: " << *matrix << std::endl;

  // set initial transform to shape model
  vtkSmartPointer< vtkTransform > transform = vtkSmartPointer< vtkTransform >::New();
  transform->SetMatrix( matrix );

  // initializer will output the transform to the shape model directly
  _shapeModel.setTransform( transform );
}

void
ShapeModelInitializer::run( double offsetX, double offsetY, double offsetZ )
{
  double centerPoint[3];
  _image.getCenter( centerPoint );
  
  // Compute the center of mass
  vtkSmartPointer< vtkCenterOfMass > centerOfMassFilter = vtkSmartPointer< vtkCenterOfMass >::New();
  centerOfMassFilter->SetInputData( _shapeModel.getPolyData() );
  centerOfMassFilter->SetUseScalarsAsWeights( false );
  centerOfMassFilter->Update();

  double center[3];
  centerOfMassFilter->GetCenter( center );

  double centerOffset[3];
  for (int k = 0; k < 3; k++)
  {
    centerOffset[k] = centerPoint[k] - center[k];
  }

  // hard coded initial rotation to reflect ShapeWorks transformation
  // turned out the rotation part is unnecessary (commented out below)
  
  // compensating translation by auto-centering and manual offset input
  vtkSmartPointer< vtkMatrix4x4 > matrix = vtkSmartPointer< vtkMatrix4x4 >::New();
  //matrix->SetElement( 0, 0, -1 );
  //matrix->SetElement( 1, 1, -1 );
  
  matrix->SetElement( 0, 3, centerOffset[0] + offsetX );
  matrix->SetElement( 1, 3, centerOffset[1] + offsetY );
  matrix->SetElement( 2, 3, centerOffset[2] + offsetZ );

  //std::cout << "Initial transform matrix: " << *matrix << std::endl;

  vtkSmartPointer< vtkTransform > transform = vtkSmartPointer< vtkTransform >::New();
  transform->SetMatrix( matrix );

  // initializer will output the transform to the shape model directly
  _shapeModel.setTransform( transform );
  
  // save transform to itk transform file for testing
  /*
  TransformFileWriterTemplateType::Pointer writer = TransformFileWriterTemplateType::New();
    
  TransformType::Pointer itkTransform = TransformType::New();
  TransformType::ParametersType parameters( Dimension * Dimension + Dimension );

  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      parameters[i * Dimension + j] = matrix->GetElement( i, j );
    }
    parameters[i + Dimension * Dimension] = matrix->GetElement( i, Dimension );
  }
  
  itkTransform->SetParameters( parameters );
  writer->SetInput( itkTransform );
  writer->SetFileName( "/Users/jinho/temp/temp.tfm" );
  writer->Update();
  */
}
