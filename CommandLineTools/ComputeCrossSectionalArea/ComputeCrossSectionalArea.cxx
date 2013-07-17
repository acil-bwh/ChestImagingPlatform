/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to compute cross sectional
 *  areas of structures in the input label map image. The
 *  cross-sectional area is computed with respect to the axial
 *  plane. The algorithm proceeds by tallying all voxels of various
 *  types in the label map. The tally for each entity is then
 *  multiplied by the in-plane (axial) spacing value to give the cross
 *  sectional areas. Quantities are printed to std out 
 *  
 *  USAGE:
 *
 *  ComputeCrossSectionalArea [-o <string>] -i <string> [--]
 *                            [--version] [-h]
 *
 *  Where:
 *
 *   -i \<string\>,  --inFileName \<string\>
 *     (required)  Input label map file name
 *
 *   -o \<string\>,  --outFileName \<string\>
 *    Output CSV file
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
 *  $Date: 2012-11-05 15:26:28 -0500 (Mon, 05 Nov 2012) $
 *  $Revision: 309 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include <iostream>
#include <fstream>

typedef itk::Image< unsigned short, 3 >       ImageType;
typedef itk::ImageFileReader< ImageType >     ReaderType;
typedef itk::ImageRegionIterator< ImageType > IteratorType;

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inFileName  = "NA";
  std::string outFileName = "NA";

  //
  // Argument descriptions for user help
  //
  std::string programDesc = "This program can be used to compute cross sectional areas \
of structures in the input label map image. The cross-sectional area is computed with \
respect to the axial plane. The algorithm proceeds by tallying all voxels of various types \
in the label map. The tally for each entity is then multiplied by the in-plane (axial) \
spacing value to give the cross sectional areas. Quantities are printed to std out";
  std::string inFileNameDesc = "Input label map file name";
  std::string outFileNameDesc = "Output CSV file";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 309 $" );

    TCLAP::ValueArg<std::string> inFileNameArg ( "i", "inFileName", inFileNameDesc, true, inFileName, "string", cl );
    TCLAP::ValueArg<std::string> outFileNameArg ( "o", "outFileName", outFileNameDesc, false, outFileName, "string", cl );
      
    cl.parse( argc, argv );

    inFileName  = inFileNameArg.getValue();
    outFileName = outFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  //
  // Instantiate conventions for later usage
  //
  ChestConventions conventions;

  //
  // Read the label map
  //
  std::cout << "Reading label map..." << std::endl;
  ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  ImageType::SpacingType spacing = reader->GetOutput()->GetSpacing();

  //
  // Get a list of the labels present in the label map 
  //
  std::cout << "Determing structures in label map..." << std::endl;
  std::list< unsigned short > labelsList;

  IteratorType it( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      labelsList.push_back( it.Get() );
      }

    ++it;
    }
  labelsList.unique();
  labelsList.sort();
  labelsList.unique();

  //
  // Now initialize the label values to counts map
  //
  std::cout << "Initializing counters..." << std::endl;
  std::map< unsigned short, unsigned int > labelValueToCountsMap;

  std::list< unsigned short >::iterator listIt;
 
  for ( listIt = labelsList.begin(); listIt != labelsList.end(); listIt++ )
    {
    labelValueToCountsMap[*listIt] = 0;
    }

  //
  // Determine the counts for the various structures
  //
  std::cout << "Counting..." << std::endl;
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() != 0 )
      {
      labelValueToCountsMap[it.Get()]++;
      }

    ++it;
    }

  std::map< unsigned short, unsigned int >::iterator mapIt;

  //
  // Writing output file (if specified), or print the cross-sectional
  // areas to the command line
  //
  if ( outFileName.compare( "NA" ) != 0 )
    { 
    std::cout<<"Writing output..."<<std::endl;
    std::ofstream writer;
    writer.open ( outFileName.c_str() );  
  
    writer << inFileName << ",";
    for ( mapIt = labelValueToCountsMap.begin(); mapIt != labelValueToCountsMap.end(); mapIt++ )
      {
      writer << conventions.GetChestRegionNameFromValue( (*mapIt).first ) << " "
             << conventions.GetChestTypeNameFromValue( (*mapIt).first ) << "(mmˆ2),";
      }  
    writer << std::endl;
  
    writer << ",";
    for ( mapIt = labelValueToCountsMap.begin(); mapIt != labelValueToCountsMap.end(); mapIt++ )
      {
      double area = spacing[0]*spacing[1]*static_cast< double >( (*mapIt).second );
      writer << area << ",";
      }
    writer << std::endl;
  
    writer.close();
    }
  else
    {
    for ( mapIt = labelValueToCountsMap.begin(); mapIt != labelValueToCountsMap.end(); mapIt++ )
      {
      double area = spacing[0]*spacing[1]*static_cast< double >( (*mapIt).second );

      std::cout << conventions.GetChestRegionNameFromValue( (*mapIt).first ) << " "
                << conventions.GetChestTypeNameFromValue( (*mapIt).first ) << ":\t" << area << " (mmˆ2)" << std::endl;
      }  
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
