/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to extract regions, types, and
 *  region-type pairs of interest from an input chest label map.
 *
 *  $Date: 2012-10-02 15:54:43 -0400 (Tue, 02 Oct 2012) $
 *  $Revision: 283 $
 *  $Author: jross $
 *
 *  USAGE: 
 *
 *  ExtractChestLabelMap  [--typePair <unsigned char>] ...  [--regionPair
 *                        <unsigned char>] ...  [-t <unsigned char>] ... 
 *                        [-r <unsigned char>] ...  -o <string> -i
 *                        <string> [--] [--version] [-h]
 *
 *  Where: 
 *
 *   --typePair <unsigned char>  (accepted multiple times)
 *     Specify a type in a region-type pair you want to extract. This
 *     flagshould be used together with the -regionPair flag
 *
 *   --regionPair <unsigned char>  (accepted multiple times)
 *     Specify a region in a region-type pair you want to extract. This
 *     flagshould be used together with the -typePair flag
 *
 *   -t <unsigned char>,  --type <unsigned char>  (accepted multiple times)
 *     Specify a type you want to extract
 *
 *   -r <unsigned char>,  --region <unsigned char>  (accepted multiple times)
 *     Specify a region you want to extract
 *
 *   -o <string>,  --outFileName <string>
 *     (required)  Output label map file name
 *
 *   -i <string>,  --inFileName <string>
 *     (required)  Input label map file name
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
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageRegionIterator.h"

typedef itk::CIPExtractChestLabelMapImageFilter       LabelMapExtractorType;
typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inFileName  = "NA";
  std::string outFileName = "NA";
  std::vector< unsigned char >  regionVec;
  std::vector< unsigned char >  typeVec;
  std::vector< unsigned char >  regionPairVec;
  std::vector< unsigned char >  typePairVec;

  //
  // Program and argument descriptions for user help
  //
  std::string programDescription = "This program can be used to extract regions, types, and region-type pairs of\
interest from an input chest label map.";
  
  std::string inFileNameDescription = "Input label map file name";
  std::string outFileNameDescription = "Output label map file name";
  std::string regionVecDescription = "Specify a region you want to extract";
  std::string typeVecDescription = "Specify a type you want to extract";
  std::string regionPairVecDescription = "Specify a region in a region-type pair you want to extract. This flag\
should be used together with the -typePair flag";
  std::string typePairVecDescription = "Specify a type in a region-type pair you want to extract. This flag\
should be used together with the -regionPair flag";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 283 $" );

    TCLAP::ValueArg<std::string>    inFileNameArg( "i", "inFileName", inFileNameDescription, true, inFileName, "string", cl );
    TCLAP::ValueArg<std::string>    outFileNameArg( "o", "outFileName", outFileNameDescription, true, outFileName, "string", cl );
    TCLAP::MultiArg<unsigned int>   regionVecArg( "r", "region", regionVecDescription, false, "unsigned int", cl );
    TCLAP::MultiArg<unsigned int>   typeVecArg( "t", "type", typeVecDescription, false, "unsigned int", cl );
    TCLAP::MultiArg<unsigned int>   regionPairVecArg( "", "regionPair", regionPairVecDescription, false, "unsigned int", cl );
    TCLAP::MultiArg<unsigned int>   typePairVecArg( "", "typePair", typePairVecDescription, false, "unsigned int", cl );

    cl.parse( argc, argv );

    inFileName    = inFileNameArg.getValue();
    outFileName   = outFileNameArg.getValue();
    for ( unsigned int i=0; i<regionVecArg.getValue().size(); i++ )
      {
      regionVec.push_back( regionVecArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<typeVecArg.getValue().size(); i++ )
      {
      typeVec.push_back( typeVecArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<regionPairVecArg.getValue().size(); i++ )
      {
      regionPairVec.push_back( regionPairVecArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<typePairVecArg.getValue().size(); i++ )
      {
      typePairVec.push_back( typePairVecArg.getValue()[i] );
      }
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  if ( regionPairVec.size() != typePairVec.size() )
    {
    std::cerr << "Error: Must specify same number of regions as types when indicating region-type pairs." << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
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

  std::cout << "Extracting..." << std::endl;
  LabelMapExtractorType::Pointer extractor = LabelMapExtractorType::New();
    extractor->SetInput( reader->GetOutput() );
  for ( unsigned int i=0; i<regionVec.size(); i++ )
    {
    extractor->SetChestRegion( regionVec[i] );
    }
  for ( unsigned int i=0; i<typeVec.size(); i++ )
    {
    extractor->SetChestType( typeVec[i] );
    }
  for ( unsigned int i=0; i<regionPairVec.size(); i++ )
    {
    extractor->SetRegionAndType( regionPairVec[i], typePairVec[i] );
    }
    extractor->Update();

  IteratorType eIt( extractor->GetOutput(), extractor->GetOutput()->GetBufferedRegion() );
  IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  eIt.GoToBegin();
  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
      rIt.Set( eIt.Get() );

      ++eIt;
      ++rIt;
    }

  std::cout << "Writing..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
  //    writer->SetInput( extractor->GetOutput() );
    writer->SetInput( reader->GetOutput() );
    writer->SetFileName( outFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;

}

#endif
