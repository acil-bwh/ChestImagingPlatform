#ifndef _ShapeModelFinalizer_h_
#define _ShapeModelFinalizer_h_

#include "ShapeModelObject.h"
#include "FitShapeModelTypes.h"
#include "PoissonRecon/PoissonRecon.h"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

class ShapeModel;
class ShapeModelImage;

class ShapeModelFinalizer : public ShapeModelObject
{
public:
  ShapeModelFinalizer( ShapeModel& shapeModel,
                       const ShapeModelImage& image,
                       const std::string& outputName,
                       const std::string& outputGeomName );
  virtual ~ShapeModelFinalizer();
  void run();
protected:
  vtkSmartPointer< vtkPolyData > convertToPolyData( PoissonRecon::MeshData& mesh );
  MeshType::Pointer convertToITKMesh( PoissonRecon::MeshData& mesh );
private:
  ShapeModel& _shapeModel;
  const ShapeModelImage& _image;
  std::string _outputName;
  std::string _outputGeomName;
};

#endif
