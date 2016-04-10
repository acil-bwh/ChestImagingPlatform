#include "ShapeModelFinalizer.h"
#include "ShapeModel.h"
#include "ShapeModelImage.h"
#include "ShapeModelMeshWriter.h"
#include "PoissonRecon/PoissonRecon.h"
#include <vtkPLYWriter.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>

ShapeModelFinalizer::ShapeModelFinalizer( ShapeModel& shapeModel,
                                          const ShapeModelImage& image,
                                          const std::string& outputName,
                                          const std::string& outputGeomName )
: _shapeModel( shapeModel ),
  _image( image ),
  _outputName( outputName ),
  _outputGeomName( outputGeomName )
{
}

ShapeModelFinalizer::~ShapeModelFinalizer()
{
}

void 
ShapeModelFinalizer::run()
{
  std::cout << "Performing Poisson surface reconstruction..." << std::endl;
  
  vtkSmartPointer< vtkPoints > vtk_points = _shapeModel.getPolyData()->GetPoints();
  
  // construct raw points from polydata
  std::vector< float > points;
  double p[3];
  for (unsigned int i = 0; i < vtk_points->GetNumberOfPoints(); i++)
  {
    vtk_points->GetPoint( i, p );
    points.push_back( p[0] );
    points.push_back( p[1] );
    points.push_back( p[2] );
  }
  
  // perform surface reconstruction from raw points
  PoissonRecon recon;
  PoissonRecon::VolumeData volume;
  PoissonRecon::MeshData& mesh = recon.createIsoSurfaceAndMesh( points, volume );
  
  std::cout << "Done. Number of triangles: " << mesh.polygonCount() << std::endl;
  
  // alternative (original) method is createBinaryMeshImage but this is better
  // since it directly creates binary image from the iso-surface volume data
  // generated during Poisson reconstruction (no more missing lines!)
  _image.createBinaryVolumeImage( volume, _outputName );
  
  // create VTK polydata to save it out output geometry file
  vtkSmartPointer< vtkPolyData > polydata = convertToPolyData( mesh );
  vtkSmartPointer< vtkPolyData > polydataModelSpace = _shapeModel.transformToModelSpace( polydata );
  ShapeModelMeshWriter::write( polydataModelSpace, _outputGeomName );
}

MeshType::Pointer 
ShapeModelFinalizer::convertToITKMesh( PoissonRecon::MeshData& mesh )
{
  MeshType::Pointer itk_mesh = MeshType::New();
  
  unsigned int num_points = mesh.outOfCorePointCount();
  mesh.resetIterator();
  
  std::cout << "Number of points to convert to ITK mesh: " << num_points << std::endl;
  
  itk_mesh->GetPoints()->Reserve( num_points );
  
  for (unsigned int i = 0; i < num_points; i++)
  {
    PlyVertex< float > v;
    if (mesh.nextOutOfCorePoint( v ))
    {
      PointType p;
      p[0] = v.point[0];
      p[1] = v.point[1];
      p[2] = v.point[2];
      
      itk_mesh->SetPoint( i, p );
    }
  }
  
  unsigned int num_faces = mesh.polygonCount();
  mesh.resetIterator();
  
  std::cout << "Number of faces to convert to ITK mesh: " << num_faces << std::endl;

  itk_mesh->GetCells()->Reserve( num_faces );
  
  typedef MeshType::CellType CellType;
  typedef itk::TriangleCell< CellType > TriangleCellType;
  
  for (unsigned int i = 0; i < num_faces; i++)
  {
    std::vector< CoredVertexIndex > vlist;
    mesh.nextPolygon( vlist );
    
    if (vlist.size() == 3) // for now, assume only triangle mesh input
    {
      MeshType::CellAutoPointer c;
      TriangleCellType* pcell = new TriangleCellType();

      pcell->SetPointId( 0, vlist[0].idx );
      pcell->SetPointId( 1, vlist[1].idx );
      pcell->SetPointId( 2, vlist[2].idx );
      
      c.TakeOwnership( pcell );
      itk_mesh->SetCell( i, c );
    }
  }
  
  return itk_mesh;
}

vtkSmartPointer< vtkPolyData > 
ShapeModelFinalizer::convertToPolyData( PoissonRecon::MeshData& mesh )
{
  vtkSmartPointer< vtkPolyData > polydata = vtkSmartPointer< vtkPolyData >::New();
  
  vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();
  
  unsigned int num_points = mesh.outOfCorePointCount();
  mesh.resetIterator();
  
  std::cout << "Number of points to convert to VTK PolyData: " << num_points << std::endl;
  
  for (unsigned int i = 0; i < num_points; i++)
  {
    PlyVertex< float > v;
    mesh.nextOutOfCorePoint( v );
    points->InsertNextPoint( v.point[0], v.point[1], v.point[2] );
  }
  
  polydata->SetPoints( points );
  
  unsigned int num_faces = mesh.polygonCount();
  mesh.resetIterator();
  
  std::cout << "Number of faces to convert to VTK PolyData: " << num_faces << std::endl;
  
  vtkSmartPointer< vtkCellArray > cells = vtkSmartPointer< vtkCellArray >::New();
  
  for (unsigned int i = 0; i < num_faces; i++)
  {
    std::vector< CoredVertexIndex > vlist;
    mesh.nextPolygon( vlist );
    
    if (vlist.size() == 3) // for now, assume only triangle mesh input
    {
      vtkSmartPointer< vtkTriangle > triangle = vtkSmartPointer< vtkTriangle >::New();
      triangle->GetPointIds()->SetId( 0, vlist[0].idx );
      triangle->GetPointIds()->SetId( 1, vlist[1].idx );
      triangle->GetPointIds()->SetId( 2, vlist[2].idx );
      cells->InsertNextCell( triangle );
    }
  }
  
  polydata->SetPolys( cells );
  
  return polydata;
}
