#include "ShapeModelInitializer.h"
#include "ShapeModel.h"
#include <vtkCenterOfMass.h>

ShapeModelInitializer::ShapeModelInitializer( ShapeModel& shapeModel,
                                              ImageType::Pointer image )
: _shapeModel(shapeModel),
  _image(image)
{
}

ShapeModelInitializer::~ShapeModelInitializer()
{
}

void
ShapeModelInitializer::run( double offsetX, double offsetY, double offsetZ )
{
  ImageType::SizeType sz = _image->GetLargestPossibleRegion().GetSize();
  ImageType::IndexType centerIndex = { sz[0]/2, sz[1]/2, sz[2]/2} ;

  ImageType::PointType centerPoint;
  _image->TransformIndexToPhysicalPoint( centerIndex, centerPoint );

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

  vtkSmartPointer< vtkTransform > transform = vtkSmartPointer< vtkTransform >::New();
  transform->SetMatrix( matrix );

  // initializer will output the transform to the shape model directly
  _shapeModel.setTransform(transform);
}
