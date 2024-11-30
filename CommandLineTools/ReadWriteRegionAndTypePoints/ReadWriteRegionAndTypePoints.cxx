/** \file
*  \ingroup commandLineTools
*  \details: This program takes as input a file of region and type points. If
*  the input file is a CSV file, it is converted to a vtk type. If the input
*  file is a vtk type, it is transformed to a CSV type
*/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "cipHelper.h"
#include "ReadWriteRegionAndTypePointsCLP.h"

namespace
{
  // Function that converts a vtk file to a csv file given the filenames as inputs
  bool ReadVTKWriteCSV(std::string vtkFileName, std::string csvFileName)
  {
    cip::ChestConventions conventions;
    
    std::cout << "Reading polydata..." << std::endl;
    vtkSmartPointer< vtkPolyDataReader > reader = vtkPolyDataReader::New();
      reader->SetFileName( vtkFileName.c_str() );
      reader->Update();

    cip::AssertChestRegionChestTypeArrayExistence( reader->GetOutput() );
    
    std::ofstream file( csvFileName.c_str() );
    double* point = new double[3];
    
    file << "Region,Type,X Location,Y Location,Z Location" << std::endl;
    for( unsigned int i=0; i<reader->GetOutput()->GetNumberOfPoints(); i++ )
      {
	reader->GetOutput()->GetPoint( i, point );
	
	unsigned short chestRegionChestTypeValue =
	  (unsigned short)(reader->GetOutput()->GetPointData()->GetArray( "ChestRegionChestType" )->GetTuple( i )[0]);
	
	file << conventions.GetChestRegionNameFromValue( chestRegionChestTypeValue ) << ",";
	file << conventions.GetChestTypeNameFromValue( chestRegionChestTypeValue ) << ",";
	file << point[0] << "," << point[1] << "," << point[2] << std::endl;
      }    
    file.close();
    
    return true;
  }
  
  // Function that converts a csv file to a vtk file given the filenames as inputs
  bool ReadCSVWriteVTK( std::string csvFileName, std::string vtkFileName )
  {
    cip::ChestConventions conventions;

    std::cout << "Reading CSV file..." << std::endl;
    cipChestRegionChestTypeLocationsIO regionTypePointsIO;
      regionTypePointsIO.SetFileName( csvFileName );
      regionTypePointsIO.Read();
    
    vtkSmartPointer< vtkDoubleArray > pointArray = vtkSmartPointer< vtkDoubleArray >::New();
      pointArray->SetNumberOfComponents( 3 );
    
    vtkSmartPointer< vtkPolyData > polyData = vtkSmartPointer< vtkPolyData >::New();
    vtkSmartPointer< vtkPoints >   points   = vtkSmartPointer< vtkPoints >::New();
    
    vtkSmartPointer< vtkFloatArray > chestRegionChestTypeArray = vtkSmartPointer< vtkFloatArray >::New();
      chestRegionChestTypeArray->SetNumberOfComponents( 1 );
      chestRegionChestTypeArray->SetName( "ChestRegionChestType" );
    
    // Get the location for each point
    for ( unsigned int i=0; i<regionTypePointsIO.GetOutput()->GetNumberOfTuples(); i++ )
      {
	cip::PointType pointLocation(3);	
	regionTypePointsIO.GetOutput()->GetLocation( i, pointLocation );
	
	double* vtkPoint = new double[3];
          vtkPoint[0] = pointLocation[0];
	  vtkPoint[1] = pointLocation[1];
	  vtkPoint[2] = pointLocation[2];
	
	unsigned char cipRegion = regionTypePointsIO.GetOutput()->GetChestRegionValue( i );
	unsigned char cipType   = regionTypePointsIO.GetOutput()->GetChestTypeValue( i );
	float chestRegionChestTypeValue = conventions.GetValueFromChestRegionAndType( cipRegion, cipType );

	chestRegionChestTypeArray->InsertTuple( i, &chestRegionChestTypeValue );
	pointArray->InsertTuple( i, vtkPoint );
      }
    
    points->SetData( pointArray );
    
    polyData->SetPoints( points );
    polyData->GetPointData()->AddArray( chestRegionChestTypeArray );
    
    std::cout << "Writing poly data..." << std::endl;
    vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
      writer->SetFileName( vtkFileName.c_str() );
      writer->SetInputData( polyData );
      writer->SetFileTypeToBinary();
      writer->Write();
    
    return true;
  }
}
int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  std::string inputExtension  = "NA";
  std::string outputExtension = "NA";
  
  // Check if the argument is a csv or a vtk file, otherwise report error
  unsigned int inputPointLoc = inputFileName.find_last_of( '.' );
  inputExtension = inputFileName.substr( inputPointLoc+1, inputFileName.length());
  
  unsigned int outputPointLoc = outputFileName.find_last_of( '.' );
  outputExtension = outputFileName.substr( outputPointLoc+1, outputFileName.length());
  
  bool readWriteSuccess = true;
  
  // If it is a csv file, invoke the reader which can be found under cip/trunk/io
  if (strcmp(inputExtension.c_str(), "csv") == 0)
    {
      // Check if the output file is the right format
      if (strcmp(outputExtension.c_str(), "vtk") != 0)
	{
	  std::cerr << "Must specify .vtk as output file format if input format is .csv" << std::endl;
	  exit(1);
	}
      
      readWriteSuccess = ReadCSVWriteVTK( inputFileName, outputFileName);
    }
  else if (strcmp(inputExtension.c_str(), "vtk") == 0)
    {
      if (strcmp(outputExtension.c_str(), "csv") != 0)
	{
	  std::cerr << "Must specify .csv as output file format if input format is .vtk" << std::endl;
	  exit(1);
	}
      
      readWriteSuccess = ReadVTKWriteCSV( inputFileName, outputFileName);
    }
  else
    {
      std::cerr << "Input file format must be .vtk or .csv" << std::endl;
      exit(1);
    }
  
  if ( readWriteSuccess )
    {
      std::cout << "DONE." << std::endl;
      return cip::EXITSUCCESS;
    }
  else
    {
      return cip::EXITFAILURE;
    }
}

#endif
