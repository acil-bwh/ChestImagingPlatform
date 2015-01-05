#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkCIPMergeChestLabelMapsImageFilter.h"
#include "itkImageRegionIterator.h"

int main( int argc, char* argv[] )
{
  typedef itk::CIPExtractChestLabelMapImageFilter< 3 >   ExtractorType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >  IteratorType;
  typedef itk::CIPMergeChestLabelMapsImageFilter         MergeType;

  cip::ChestConventions conventions;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New();
    reader->SetFileName( argv[1] );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading test image:";
    std::cerr << excp << std::endl;
    }

  // First test: test the 'Union' option
  {
    std::cout << "Extracting whole lung..." << std::endl;
    ExtractorType::Pointer wholeLungExtractor = ExtractorType::New();
      wholeLungExtractor->SetInput( reader->GetOutput() );
      wholeLungExtractor->SetChestRegion( 1 );
      wholeLungExtractor->Update();

    std::cout << "Extracting right lung and airway..." << std::endl;
    ExtractorType::Pointer rightLungAirwayExtractor = ExtractorType::New();
      rightLungAirwayExtractor->SetInput( reader->GetOutput() );
      rightLungAirwayExtractor->SetChestRegion( 2 );
      rightLungAirwayExtractor->SetChestType( 2 );
      rightLungAirwayExtractor->SetRegionAndType( 2, 2 );
      rightLungAirwayExtractor->Update();
    
    std::cout << "Merging..." << std::endl;
    MergeType::Pointer merger = MergeType::New();
      merger->SetOverlayImage( wholeLungExtractor->GetOutput() );
      merger->SetInput( rightLungAirwayExtractor->GetOutput() );
      merger->SetUnion( true );
      merger->Update();

    IteratorType mIt( merger->GetOutput(), merger->GetOutput()->GetBufferedRegion() );
    IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );
    
    mIt.GoToBegin();
    rIt.GoToBegin();
    while ( !mIt.IsAtEnd() )
      {
	unsigned char mergedRegion = conventions.GetChestRegionFromValue( mIt.Get() );
	unsigned char mergedType   = conventions.GetChestTypeFromValue( mIt.Get() );

	unsigned char inRegion = conventions.GetChestRegionFromValue( rIt.Get() );
	unsigned char inType   = conventions.GetChestTypeFromValue( rIt.Get() );

    	if ( (inRegion == (unsigned char)(cip::AIRWAY) && mergedRegion != (unsigned char)(cip::AIRWAY)) ||
	     (inRegion != (unsigned char)(cip::AIRWAY) && mergedRegion == (unsigned char)(cip::AIRWAY)) )
    	  {
	    std::cout << "FAILED" << std::endl;
	    return 1;
    	  }

    	if ( inRegion == (unsigned char)(cip::LEFTLUNG) && mergedRegion != (unsigned char)(cip::WHOLELUNG) )
    	  {
	    std::cout << "FAILED" << std::endl;
	    return 1;
    	  }
    	
    	if ( inRegion == (unsigned char)(cip::RIGHTLUNG) && mergedRegion != (unsigned char)(cip::RIGHTLUNG) )
    	  {
	    std::cout << "FAILED" << std::endl;
	    return 1;
    	  }

    	++mIt;
    	++rIt;
      }
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
