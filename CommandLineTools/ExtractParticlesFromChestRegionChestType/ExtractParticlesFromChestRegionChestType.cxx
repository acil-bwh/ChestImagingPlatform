/** \file
 *  \ingroup commandLineTools 
 *  \details This program allows you to extract particles from an
 *  input particles data set using either an input label map or the
 *  particles themselves. Many particles datasets contain a
 *  'ChestType' and 'ChestRegion' data field. The values of these
 *  fields can be used to isolate particles of
 *  interest. Alternatively, a label map can be specified, and only
 *  the particles falling in the specified region of interest will be
 *  written. Additionally, even if the input particles do not have the
 *  'ChestType' or 'ChestRegion' data arrays, the output particles
 *  data set will have these with region and type values specified at
 *  the command line.
 * 
 *  $Date: 2013-03-25 13:23:52 -0400 (Mon, 25 Mar 2013) $
 *  $Revision: 383 $
 *  $Author: jross $
 *
 *  USAGE: 
 *
 *  ExtractParticlesFromChestRegionChestType  [-t \<int\>] -r \<int\> 
 *                                            -o \<string\> -i \<string\>
 *                                            [-l \<string\>]  
 *                                            [--] [--version] [-h]
 *
 *  Where: 
 *
 *   -t \<int\>,  --type \<int\>
 *     Chest type for which to extract particles. If specifying a label map
 *     this flag shouldbe used to indicate the type of particles in the input
 *     file for output array labeling purposes (if no value is
 *     specifiedUNDEFINEDTYPE will be set as the particle ChestType field
 *     value
 *
 *   -r \<int\>,  --region \<int\>
 *     (required)  Chest region from which to extract particles
 *
 *   -o \<string\>,  --outParticles \<string\>
 *     (required)  Output particles file name
 *
 *   -i \<string\>,  --inParticles \<string\>
 *     (required)  Input particles file name
 *
 *   -l \<string\>,  --labelMap \<string\>
 *     Input label map file name. This is an optional input. If no label map
 *     is specified,the 'ChestRegion' and 'ChestType' arrays in the input
 *     will be used to extract theregion or type specified with the '-r' and
 *     '-t' flags, respectively
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
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipConventions.h"
#include "cipHelper.h"

void GetOutputParticlesUsingLabelMap( std::string, std::vector< unsigned char >, vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );
void GetOutputParticlesUsingChestRegionChestTypeArrays( std::vector< unsigned char >, std::vector< unsigned char >, 
							vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string labelMapFileName     = "NA";
  std::string inParticlesFileName  = "NA";
  std::string outParticlesFileName = "NA";
  std::vector< unsigned char > cipRegions;
  std::vector< unsigned char > cipTypes;

  // Program and argument descriptions for user help
  std::string programDescription = "This program allows you to extract particles from an input particles data set using either\
an input label map or the particles themselves. Many particles datasets contain a 'ChestType' and 'ChestRegion' data field.\
The values of these fields can be used to isolate particles of interest. Alternatively, a label map can be specified, and only\
the particles falling in the specified region of interest will be written. Additionally, even if the input particles do not have\
the ChestType' or 'ChestRegion' data arrays, the output particles data set will have these with region and type values specified\
at the command line.";

  std::string labelMapFileNameDescription     = "Input label map file name. This is an optional input. If no label map is specified,\
the 'ChestRegion' and 'ChestType' arrays in the input will be used to extract the\
region or type specified with the '-r' and '-t' flags, respectively";
  std::string inParticlesFileNameDescription  = "Input particles file name";
  std::string outParticlesFileNameDescription = "Output particles file name";
  std::string cipRegionsDescription            = "Chest regions from which to extract particles";
  std::string cipTypesDescription              = "Chest types for which to extract particles. If specifying a label map this flag \
is not relevent.";

  // Parse the input arguments
  try
    {
    TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 383 $" );

    TCLAP::ValueArg<std::string> labelMapFileNameArg( "l", "labelMap", labelMapFileNameDescription, false, labelMapFileName, "string", cl );
    TCLAP::ValueArg<std::string> inParticlesFileNameArg( "i", "inParticles", inParticlesFileNameDescription, true, inParticlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> outParticlesFileNameArg( "o", "outParticles", outParticlesFileNameDescription, true, outParticlesFileName, "string", cl );
    TCLAP::MultiArg<unsigned char> cipRegionsArg( "r", "region", cipRegionsDescription, false, "unsigned char", cl );
    TCLAP::MultiArg<unsigned char> cipTypesArg( "t", "type", cipTypesDescription, false, "unsigned char", cl );

    cl.parse( argc, argv );

    labelMapFileName     = labelMapFileNameArg.getValue();
    inParticlesFileName  = inParticlesFileNameArg.getValue();
    outParticlesFileName = outParticlesFileNameArg.getValue();

    if ( cipRegionsArg.getValue().size() == 0 )
      {
	cipRegions.push_back( (unsigned char)(cip::UNDEFINEDREGION) );
      }
    for ( unsigned int i=0; i<cipRegionsArg.getValue().size(); i++ )
      {
	cipRegions.push_back( (unsigned char)(cipRegionsArg.getValue()[i]) );
      }
    for ( unsigned int i=0; i<cipTypesArg.getValue().size(); i++ )
      {
	cipTypes.push_back( (unsigned char)(cipTypesArg.getValue()[i]) );
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  vtkSmartPointer< vtkPolyData > outParticles = vtkSmartPointer< vtkPolyData >::New();

  std::cout << "Reading polydata..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
    particlesReader->SetFileName( inParticlesFileName.c_str() );
    particlesReader->Update();    

  // First make sure that the particles have 'ChestRegion' and 'ChestType' arrays
  cip::AssertChestRegionChestTypeArrayExistence( particlesReader->GetOutput() );

  if ( labelMapFileName.compare( "NA" ) != 0 )
    {
    GetOutputParticlesUsingLabelMap( labelMapFileName, cipRegions, particlesReader->GetOutput(), outParticles );
    }
  else 
    {
    GetOutputParticlesUsingChestRegionChestTypeArrays( cipRegions, cipTypes, particlesReader->GetOutput(), outParticles );
    }

  std::cout << "Writing extracted particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
    particlesWriter->SetFileName( outParticlesFileName.c_str() );
    particlesWriter->SetInput( outParticles );
    particlesWriter->Write();

  std::cout << "DONE." << std::endl;

  return 0;
}

// A) If no region is specified, all particles falling within the foreground 
// region will be retained. The particles' ChestRegion and ChestType array values
// will be preserved. 
// B) If a region is specified, all particles falling inside the specified label
// map's region will be retained. The particles' ChestType array is preserved, but
// the ChestRegion is overwritten with the specified desired region 
void GetOutputParticlesUsingLabelMap( std::string fileName, std::vector< unsigned char > cipRegions, 
				      vtkSmartPointer< vtkPolyData > inParticles, vtkSmartPointer< vtkPolyData > outParticles )
{
  cip::ChestConventions conventions;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( fileName );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  unsigned int numberPointDataArrays = inParticles->GetPointData()->GetNumberOfArrays();
  unsigned int numberParticles       = inParticles->GetNumberOfPoints();

  vtkSmartPointer< vtkPoints > outputPoints  = vtkSmartPointer< vtkPoints >::New();
  
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<numberPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inParticles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );
    }

  unsigned int inc = 0;
  cip::LabelMapType::PointType point;
  cip::LabelMapType::IndexType index;

  for ( unsigned int i=0; i<numberParticles; i++ )
    {
    point[0] = inParticles->GetPoint(i)[0];
    point[1] = inParticles->GetPoint(i)[1];
    point[2] = inParticles->GetPoint(i)[2];
    
    labelMapReader->GetOutput()->TransformPhysicalPointToIndex( point, index );    

    unsigned short labelValue   = labelMapReader->GetOutput()->GetPixel( index );
    unsigned char  labelRegion  = conventions.GetChestRegionFromValue( labelValue );
    
    if ( labelValue > 0 )
      {       
	for ( unsigned int k=0; k<cipRegions.size(); k++ )
	  {
	    unsigned char cipRegion = cipRegions[k];

	    // If the label map chest region is a subordinate of the requested
	    // chest region, then add this particle to the output    
	    if ( cipRegion == (unsigned char)(cip::UNDEFINEDREGION ) || 
		 conventions.CheckSubordinateSuperiorChestRegionRelationship( labelRegion, cipRegion ) )
	      {
		outputPoints->InsertNextPoint( inParticles->GetPoint(i) );
		
		for ( unsigned int j=0; j<numberPointDataArrays; j++ )
		  {
		    arrayVec[j]->InsertTuple( inc, inParticles->GetPointData()->GetArray(j)->GetTuple(i) );        
		  }
	    
		inc++;

		break;
	      }
	  }
      }
    }
  
  outParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<numberPointDataArrays; j++ )
    {
    outParticles->GetPointData()->AddArray( arrayVec[j] );
    }
}


void GetOutputParticlesUsingChestRegionChestTypeArrays( std::vector< unsigned char > cipRegions, std::vector< unsigned char > cipTypes, 
							vtkSmartPointer< vtkPolyData > inParticles, vtkSmartPointer< vtkPolyData > outParticles )
{
  unsigned int numberPointDataArrays = inParticles->GetPointData()->GetNumberOfArrays();
  unsigned int numberParticles       = inParticles->GetNumberOfPoints();

  vtkSmartPointer< vtkPoints > outputPoints  = vtkSmartPointer< vtkPoints >::New();
  
  std::vector< vtkSmartPointer< vtkFloatArray > > pointArrayVec;

  for ( unsigned int i=0; i<numberPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inParticles->GetPointData()->GetArray(i)->GetName() );

    pointArrayVec.push_back( array );
    }

  unsigned int inc = 0;
  for ( unsigned int i=0; i<numberParticles; i++ )
    {
      for ( unsigned int j=0; j<cipTypes.size(); j++ )
	{
	  if ( static_cast< int >( inParticles->GetPointData()->GetArray( "ChestType" )->GetTuple(i)[0] ) == cipTypes[j] )
	    {
	      outputPoints->InsertNextPoint( inParticles->GetPoint(i) );

	      for ( unsigned int j=0; j<numberPointDataArrays; j++ )
		{
		  pointArrayVec[j]->InsertTuple( inc, inParticles->GetPointData()->GetArray(j)->GetTuple(i) );
		}

	      inc++;
	    }
	}
    }

  outParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<numberPointDataArrays; j++ )
    {
    outParticles->GetPointData()->AddArray( pointArrayVec[j] );
    }
}


#endif
