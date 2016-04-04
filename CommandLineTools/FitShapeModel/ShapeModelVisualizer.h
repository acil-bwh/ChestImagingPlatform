#ifndef _ShapeModelVisualizer_h_
#define _ShapeModelVisualizer_h_

#include "FitShapeModelTypes.h"
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

class ShapeModel;

class ShapeModelVisualizer
{
public:
  ShapeModelVisualizer( const ShapeModel& shapeModel,
                        MeshType::Pointer mesh,
                        ImageType::Pointer image,
                        const std::string& outputName,
                        const std::string& outputGeomName );
  virtual ~ShapeModelVisualizer();
  void setMesh( MeshType::Pointer mesh ) { _mesh = mesh; }
  void update( double sigma, int iteration = 0 );
  void writeVTKPointsToOBJ( vtkSmartPointer< vtkPoints > points, int iteration );

protected:
  void updateMeshPoints( vtkSmartPointer< vtkPoints > points );
  void createGradientMagnitudeImage( double sigma );

private:
  const ShapeModel& _shapeModel;
  MeshType::Pointer _mesh;
  ImageType::Pointer _image;
  std::string _outputName;
  std::string _outputGeomName;
  ImageType::Pointer _gradientMagnitudeImage;
};

#endif
