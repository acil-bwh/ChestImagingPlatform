#include "ShapeModelVisualizer.h"
#include "ShapeModel.h"
#include "VNLVTKConverters.h"
#include "VTKPolyDataToITKMesh.h"
#include "ShapeModelMeshWriter.h"
#include "ShapeModelUtils.h"

#include <vnl/vnl_inverse.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkXMLPolyDataWriter.h>
#include <climits>
#include <sstream>

ShapeModelVisualizer::ShapeModelVisualizer( const ShapeModel& shapeModel,
                                            const ShapeModelImage& image,
                                            const std::string& outputName,
                                            const std::string& outputGeomName,
                                            bool outputGeomInModelSpace )
: _shapeModel( shapeModel ),
  _image( image ),
  _outputName( outputName ),
  _outputGeomName( outputGeomName ),
  _outputGeomInModelSpace( outputGeomInModelSpace )
{
  _mesh = VTKPolyDataToITKMesh::convert( _shapeModel.getPolyData() );
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
    MeshType::PointType& p = pit.Value();
    p[0] = vp[0]; p[1] = vp[1]; p[2] = vp[2];
  }
}

void
ShapeModelVisualizer::update( double sigma, int iteration /*=0*/)
{
  updateMeshPoints( _shapeModel.getPolyData()->GetPoints() );

  std::cout << "Creating binary image from mesh, writing to " << _outputName << std::endl;
  
  // using ITK filter: TriangleMeshToBinaryImageFilter
  _image.createBinaryMeshImage( _mesh, _outputName );
  
  // this flag is mostly for debugging purpose and it takes some time
  // for a larger size of input images
  bool useGradientMagnitudeImage = false;

  if (useGradientMagnitudeImage) // use Gaussian gradient magnitude image for overlay output
  {
    _image.createGradientMagnitudeImage( sigma, _outputName );
  }

  // Write output geometry
  if (!_outputGeomName.empty())
  {
    vtkSmartPointer< vtkPolyData > polydata =
      (_outputGeomInModelSpace) ? _shapeModel.getPolyDataModelSpace()
                                : _shapeModel.getPolyData();
    ShapeModelMeshWriter::write( polydata, _outputGeomName );
  }
}
