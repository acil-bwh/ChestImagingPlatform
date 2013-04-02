/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to scour an input particles data
 *  set, removing specified field data arrays.
 *
 *  USAGE:
 *
 *  ScourParticleData  [-n <string>] -o <string> -i <string> [--]
 *                     [--version] [-h] 
 *
 *  Where:
 *
 *   -n <string>,  --name <string>  (accepted multiple times)
 *     Specify the name(s) of the arrays to be removed
 *
 *   -o <string>,  --out <string>
 *     (required)  Output particles file name
 *
 *   -i <string>,  --in <string>
 *     (required)  Input particles file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date: 2012-06-20 07:09:53 -0700 (Wed, 20 Jun 2012) $
 *  $Revision: 184 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "cipConventions.h"


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inParticlesFileName  = "NA";
  std::string outParticlesFileName = "NA";
  std::vector< std::string > dataArrayNames;

  //
  // Program and argument descriptions for user help
  //
  std::string programDesc = "This program can be used to scour an input particles data set, \
removing specified field data arrays.";

  std::string inParticlesFileNameDesc  = "Input particles file name";
  std::string outParticlesFileNameDesc = "Output particles file name";
  std::string dataArrayNamesDesc       = "Specify the name(s) of the arrays to be removed";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 184 $" );

    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "in", inParticlesFileNameDesc, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> outParticlesFileNameArg( "o", "out", outParticlesFileNameDesc, true, outParticlesFileName, "string", cl );
    TCLAP::MultiArg<std::string> dataArrayNamesArg( "n", "name", dataArrayNamesDesc, false, "string", cl );

    cl.parse( argc, argv );

    inParticlesFileName    = inParticlesFileNameArg.getValue();
    outParticlesFileName   = outParticlesFileNameArg.getValue();
    for ( unsigned int i=0; i<dataArrayNamesArg.getValue().size(); i++ )
      {
      dataArrayNames.push_back( dataArrayNamesArg.getValue()[i] );
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading particles ..." << std::endl;
  vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName( inParticlesFileName.c_str() );
    reader->Update();

  unsigned int numberOfFieldDataArrays = reader->GetOutput()->GetFieldData()->GetNumberOfArrays();
  unsigned int numberInputParticles    = reader->GetOutput()->GetNumberOfPoints();

  vtkPoints* points  = vtkPoints::New();
  std::vector< vtkFloatArray* > arrayVec;

  //
  // This map will keep track of the mapping between in the input
  // arrays and output arrays
  //
  std::map< unsigned int, unsigned int > outArrayToInArrayMap;

  std::cout << "Scouring..." << std::endl;
  unsigned int inc = 0;
  for ( unsigned int i=0; i<numberOfFieldDataArrays; i++ )
    {
    std::string name( reader->GetOutput()->GetFieldData()->GetArray(i)->GetName() );

    //
    // Determine if this array is one that needs to be removed
    //
    bool remove = false;
    for ( unsigned int r=0; r<dataArrayNames.size(); r++ )
      {
      if ( name.compare( dataArrayNames[r] ) == 0 )
        {
        remove = true;
        break;
        }
      }

    if ( !remove )
      {
      vtkFloatArray* array = vtkFloatArray::New();
        array->SetNumberOfComponents( reader->GetOutput()->GetFieldData()->GetArray(i)->GetNumberOfComponents() );
        array->SetName( reader->GetOutput()->GetFieldData()->GetArray(i)->GetName() );

      arrayVec.push_back( array );

      outArrayToInArrayMap[inc] = i;
      inc++;
      }
    } 

  inc = 0;
  for ( unsigned int p=0; p<numberInputParticles; p++ )
    {
    points->InsertNextPoint( reader->GetOutput()->GetPoint(p) );
    for ( unsigned int k=0; k<arrayVec.size(); k++ )
      {
      arrayVec[k]->InsertTuple( inc, reader->GetOutput()->GetFieldData()->GetArray(outArrayToInArrayMap[k])->GetTuple(p) );
      }
    inc++;
    }

  vtkPolyData* outParticles = vtkPolyData::New();
    outParticles->SetPoints( points );
  for ( unsigned int k=0; k<arrayVec.size(); k++ )
    {
    outParticles->GetFieldData()->AddArray( arrayVec[k] );
    }

  std::cout << "Writing filtered particles ..." << std::endl;
  vtkPolyDataWriter *filteredWriter = vtkPolyDataWriter::New();
    filteredWriter->SetFileName( outParticlesFileName.c_str() );
    filteredWriter->SetInput( outParticles );
    filteredWriter->Write();  

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


#endif
