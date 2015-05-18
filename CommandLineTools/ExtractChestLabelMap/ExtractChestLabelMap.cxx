/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to extract regions, types, and
 *  region-type pairs of interest from an input chest label map.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageRegionIterator.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "ExtractChestLabelMapCLP.h"

typedef itk::CIPExtractChestLabelMapImageFilter< 3 >  LabelMapExtractorType;
typedef itk::ImageRegionIterator< cip::LabelMapType > IteratorType;

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

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
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( regionVec[i] );
    extractor->SetChestRegion( cipRegion );
    }
  for ( unsigned int i=0; i<typeVec.size(); i++ )
    {
    unsigned char cipType = conventions.GetChestTypeValueFromName( typeVec[i] );
    extractor->SetChestType( cipType );
    }
  for ( unsigned int i=0; i<regionPairVec.size(); i++ )
    {
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( regionPairVec[i] );
    unsigned char cipType   = conventions.GetChestTypeValueFromName( typePairVec[i] );
    extractor->SetRegionAndType( cipRegion, cipType );
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
