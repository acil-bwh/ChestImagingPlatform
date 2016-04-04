#ifndef _ShapeModelFinalizer_h_
#define _ShapeModelFinalizer_h_

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include "FitShapeModelTypes.h"
#include "PoissonRecon/PoissonRecon.h"

class ShapeModel;

class ShapeModelFinalizer
{
public:
  ShapeModelFinalizer( ShapeModel& shapeModel );
  virtual ~ShapeModelFinalizer();
  MeshType::Pointer getMesh() const { return _itkMesh; }
  void run();
protected:
  vtkSmartPointer< vtkPolyData > convertToPolyData( PoissonRecon::MeshData& mesh );
  MeshType::Pointer convertToITKMesh( PoissonRecon::MeshData& mesh );
private:
  ShapeModel& _shapeModel;
  MeshType::Pointer _itkMesh;
};

#endif
