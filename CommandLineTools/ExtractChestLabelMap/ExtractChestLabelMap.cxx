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

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageRegionIterator.h"
#include "ExtractChestLabelMapCLP.h"

typedef itk::CIPExtractChestLabelMapImageFilter       LabelMapExtractorType;
typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

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
    extractor->SetChestRegion( (unsigned char)(regionVec[i]) );
    }
  for ( unsigned int i=0; i<typeVec.size(); i++ )
    {
    extractor->SetChestType( (unsigned char)(typeVec[i]) );
    }
  for ( unsigned int i=0; i<regionPairVec.size(); i++ )
    {
    extractor->SetRegionAndType( (unsigned char)(regionPairVec[i]), (unsigned char)(typePairVec[i]) );
    }
    extractor->Update();

  std::cout << "Writing..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( extractor->GetOutput() );
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
