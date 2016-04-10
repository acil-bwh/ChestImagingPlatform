#ifndef _ShapeModelMeshWriter_h_
#define _ShapeModelMeshWriter_h_

#include "VTKPolyDataToITKMesh.h"
#include "ShapeModelUtils.h"
#include <vtkPLYWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <string.h>

class ShapeModelMeshWriter
{
public:
  static void write( vtkSmartPointer< vtkPolyData > polydata, 
                     const std::string& outputName )
  {
    if (outputName.empty())
    {
      return;
    }
    std::cout << "Writing " << outputName << std::endl;
    
    std::string ext = outputName.substr( outputName.find_last_of('.') + 1 );
    if (iequals( ext, "PLY" ))
    {
      vtkSmartPointer< vtkPLYWriter > plyWriter = vtkSmartPointer< vtkPLYWriter >::New();
      plyWriter->SetFileName( outputName.c_str() );
      plyWriter->SetInputData( polydata );
      plyWriter->Update();
    }
    else if (iequals( ext, "OBJ" )) // VTK does not have vtkOBJWriter...
    {
      // convert vtk PolyData to itk Mesh to use itk MeshWriter
      MeshType::Pointer itkMesh = VTKPolyDataToITKMesh::convert( polydata );
      MeshWriterType::Pointer meshWriter = MeshWriterType::New();
      meshWriter->SetFileName( outputName.c_str() );
      meshWriter->SetInput( itkMesh );
      try
      {
        meshWriter->Update();
      }
      catch (itk::ExceptionObject& e)
      {
        throw std::runtime_error( e.what() );
      }
    }
    else if (iequals( ext, "PNT" ) || 
             iequals( ext, "PTS" ) ||
             iequals( ext, "TXT" ) ||
             iequals( ext, "ASC" )) // ascii point file
    {
      std::ofstream ofs( outputName.c_str() );
      vtkSmartPointer< vtkPoints > points = polydata->GetPoints();
      for (unsigned int i = 0; i < points->GetNumberOfPoints(); i++)
      {
        double p[3];
        points->GetPoint( i, p );
        ofs << p[0] << " " << p[1] << " " << p[2] << std::endl;
      }
      ofs.close();
    }
    else // default VTK XML (including no extension in Slicer UI)
    {
      if ( ext.empty() )
      {
        std::cout << "No file extension. ";
      }
      std::cout << "Writing in VTK XML format..." << std::endl;
      vtkSmartPointer< vtkXMLPolyDataWriter > polydataWriter = vtkSmartPointer< vtkXMLPolyDataWriter >::New();
      polydataWriter->SetFileName( outputName.c_str() );
      polydataWriter->SetInputData( polydata );
      polydataWriter->Write();
    }
  }
};

#endif //_ShapeModelMeshWriter_h_