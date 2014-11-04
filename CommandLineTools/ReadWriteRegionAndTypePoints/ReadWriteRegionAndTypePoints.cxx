/** \file
*  \ingroup commandLineTools
*  \details: This program takes as input a file of region and type points. If
*  the input file is a CSV file, it is converted to a vtk type. If the input
*  file is a vtk type, it is transformed to a CSV type
*
*  USAGE:
*
*  ReadWriteRegionAndTypePoints   -i <string> -o <string> [--] [--version][-h]
*
*  Where:
*
*  -i <string>,  --input <string>
*    (required)  Input file name
*
*  -o <string>,  --output <string>
*    (required)  Output imge file name
*
*  --,  --ignore_rest
*    Ignores the rest of the labeled arguments following this flag.
*
*  --version
*    Displays version information and exits.
*
*  -h,  --help
*    Displays usage information and exits.
*
*  Known bugs: The readers for both file types do not output an error when the files do not exist
*
*  $Date: 2013-04-01 16:23:05 -0400 (Mon, 01 Apr 2013) $
*  $Revision: 397 $
*  $Author: jross $
*
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
#include "ReadWriteRegionAndTypePointsCLP.h"

namespace
{
    //
    // Function that converts a vtk file to a csv file given the filenames as inputs
    //
    bool ReadVTKWriteCSV(std::string vtkFileName, std::string csvFileName)
    {
        cip::ChestConventions conventions;

        std::cout << "Reading polydata..." << std::endl;
        vtkSmartPointer< vtkPolyDataReader > reader = vtkPolyDataReader::New();
        reader->SetFileName( vtkFileName.c_str() );
        reader->Update();

        std::ofstream file( csvFileName.c_str() );
        double* point = new double[3];

        file << "Region,Type,X Location,Y Location,Z Location" << std::endl;
        for( unsigned int i=0; i<reader->GetOutput()->GetNumberOfPoints(); i++ )
        {
            reader->GetOutput()->GetPoint( i, point );

            unsigned char cipRegion =
            (unsigned char)(reader->GetOutput()->GetPointData()->GetArray( "ChestRegion" )->GetTuple( i )[0]);
            unsigned char cipType =
            (unsigned char)(reader->GetOutput()->GetPointData()->GetArray( "ChestType" )->GetTuple( i )[0]);

            file << conventions.GetChestRegionNameFromValue( cipRegion ) << ",";
            file << conventions.GetChestTypeNameFromValue( cipType ) << ",";
            file << point[0] << "," << point[1] << "," << point[2] << std::endl;
        }

        file.close();

        return true;
    }

    //
    // Function that converts a csv file to a vtk file given the filenames as inputs
    //
    bool ReadCSVWriteVTK( std::string csvFileName, std::string vtkFileName )
    {
        std::cout << "Reading CSV file..." << std::endl;
        cipChestRegionChestTypeLocationsIO regionTypePointsIO;
        regionTypePointsIO.SetFileName( csvFileName );
        regionTypePointsIO.Read();

        vtkSmartPointer< vtkDoubleArray > pointArray = vtkSmartPointer< vtkDoubleArray >::New();
        pointArray->SetNumberOfComponents( 3 );

        vtkSmartPointer< vtkPolyData > polyData = vtkSmartPointer< vtkPolyData >::New();
        vtkSmartPointer< vtkPoints >   points   = vtkSmartPointer< vtkPoints >::New();

        vtkSmartPointer< vtkFloatArray > cipRegionArray = vtkSmartPointer< vtkFloatArray >::New();
        cipRegionArray->SetNumberOfComponents( 1 );
        cipRegionArray->SetName( "ChestRegion" );

        vtkSmartPointer< vtkFloatArray > cipTypeArray = vtkSmartPointer< vtkFloatArray >::New();
        cipTypeArray->SetNumberOfComponents( 1 );
        cipTypeArray->SetName( "ChestType" );

        // Get the location for each point
        for ( unsigned int i=0; i<regionTypePointsIO.GetOutput()->GetNumberOfTuples(); i++ )
        {
            double* pointLocation = new double[3];

            regionTypePointsIO.GetOutput()->GetLocation( i, pointLocation );

            float cipRegion = float( regionTypePointsIO.GetOutput()->GetChestRegionValue( i ) );
            float cipType   = float( regionTypePointsIO.GetOutput()->GetChestTypeValue( i ) );

            cipTypeArray->InsertTuple( i, &cipType );
            cipRegionArray->InsertTuple( i, &cipRegion );
            pointArray->InsertTuple( i, pointLocation );
        }

        points->SetData( pointArray );

        polyData->SetPoints( points );
        polyData->GetPointData()->AddArray( cipRegionArray );
        polyData->GetPointData()->AddArray( cipTypeArray );

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
