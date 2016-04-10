#ifndef _ShapeModelVisualizer_h_
#define _ShapeModelVisualizer_h_

#include "FitShapeModelTypes.h"
#include "ShapeModelObject.h"
#include "ShapeModelImage.h"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

class ShapeModel;

class ShapeModelVisualizer : public ShapeModelObject
{
public:
  ShapeModelVisualizer( const ShapeModel& shapeModel,
                        const ShapeModelImage& image,
                        const std::string& outputName,
                        const std::string& outputGeomName,
                        bool outputGeomInModelSpace );
  virtual ~ShapeModelVisualizer();
  void update( double sigma, int iteration = 0 );

protected:
  void updateMeshPoints( vtkSmartPointer< vtkPoints > points );

private:
  const ShapeModel& _shapeModel;
  const ShapeModelImage& _image;
  std::string _outputName;
  std::string _outputGeomName;
  MeshType::Pointer _mesh;
  bool _outputGeomInModelSpace;
};

#endif
