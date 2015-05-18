#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "itkCIPMergeChestLabelMapsImageFilter.h"
#include "cipHelper.h"
#include "MergeChestLabelMapsCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

  std::cout << "Reading base label map..." << std::endl;
  cip::LabelMapReaderType::Pointer baseReader = cip::LabelMapReaderType::New();
    baseReader->SetFileName( baseLabelMapFileName );
  try
    {
    baseReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading base label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Reading overlay label map..." << std::endl;
  cip::LabelMapReaderType::Pointer overlayReader = cip::LabelMapReaderType::New();
    overlayReader->SetFileName( overlayLabelMapFileName );
  try
    {
    overlayReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading base label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Merging..." << std::endl;
  itk::CIPMergeChestLabelMapsImageFilter::Pointer merger = itk::CIPMergeChestLabelMapsImageFilter::New();
    merger->SetInput( baseReader->GetOutput() );
    merger->SetOverlayImage( overlayReader->GetOutput() );
  if ( applyUnion )
    {
    merger->SetUnion( true );
    }
  for ( unsigned int i=0; i<overrideTypes.size(); i++ )
    {
    unsigned char cipType = conventions.GetChestTypeValueFromName( overrideTypes[i] );
    merger->SetOverrideChestType( cipType );
    }
  for ( unsigned int i=0; i<overrideRegions.size(); i++ )
    {
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( overrideRegions[i] );
    merger->SetOverrideChestRegion( cipRegion );
    }
  for ( unsigned int i=0; i<overrideRegionTypePairs.size(); i+=2 )
    {
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( overrideRegionTypePairs[i] );
    unsigned char cipType   = conventions.GetChestTypeValueFromName( overrideRegionTypePairs[i+1] );
    merger->SetOverrideChestRegionTypePair( cipRegion, cipType );
    }
  for ( unsigned int i=0; i<mergeTypes.size(); i++ )
    {
    unsigned char cipType = conventions.GetChestTypeValueFromName( mergeTypes[i] );
    merger->SetMergeChestType( cipType );
    }
  for ( unsigned int i=0; i<mergeRegions.size(); i++ )
    {
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( overrideRegions[i] );
    merger->SetMergeChestRegion( cipRegion );
    }
  for ( unsigned int i=0; i<mergeRegionTypePairs.size(); i+=2 )
    {
    unsigned char cipRegion = conventions.GetChestRegionValueFromName( overrideRegionTypePairs[i] );
    unsigned char cipType   = conventions.GetChestTypeValueFromName( overrideRegionTypePairs[i+1] );
    merger->SetMergeChestRegionTypePair( cipRegion, cipType );
    }
    merger->Update();

  std::cout << "Writing merged label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
    writer->SetInput( merger->GetOutput() );
    writer->SetFileName( outLabelMapFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return 0;
}


#endif
