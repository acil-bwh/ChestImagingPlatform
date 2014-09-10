#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIterator.h"
#include "cipExceptionObject.h"

int main( int argc, char* argv[] )
{
  typedef itk::ImageRegionIterator< cip::LabelMapType >  IteratorType;

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

  // First test: try to close a region that does not exist. This should not
  // do anything, so confirm that the output is the same as the input
  {
    cip::LabelMapType::Pointer inputCopy = cip::LabelMapType::New();
      inputCopy->SetRegions( reader->GetOutput()->GetBufferedRegion().GetSize() );
      inputCopy->Allocate();
    
    IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );
    IteratorType cIt( inputCopy, inputCopy->GetBufferedRegion() );
    
    rIt.GoToBegin();
    cIt.GoToBegin();
    while ( !cIt.IsAtEnd() )
      {
	cIt.Set( rIt.Get() );
	
	++rIt;
	++cIt;
      }
    
    std::cout << "Closing..." << std::endl;
    try
      {
	cip::CloseLabelMap( inputCopy, (unsigned char)(cip::UPPERTHIRD), (unsigned char)(cip::UNDEFINEDTYPE), 1, 1, 1 );
      }
    catch ( cip::ExceptionObject &excp )
      {
	std::cerr << "Exception caught closing:";
	std::cerr << excp << std::endl;
      }

    IteratorType closeIt( inputCopy, inputCopy->GetBufferedRegion() );
    
    rIt.GoToBegin();
    closeIt.GoToBegin();
    while ( !closeIt.IsAtEnd() )
      {
	if ( closeIt.Get() != rIt.Get() )
	  {
	    std::cout << "FAILED" << std::endl;
	    return 1;
	  }
	
	++rIt;
	++closeIt;
      }
  }

  // Second test: try to get the bounding box of a region that does not exist in
  // the label map. The returned region should have zero size.
  {
    std::cout << "Getting bouding box..." << std::endl;
    cip::LabelMapType::RegionType bbRegion = 
      cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion( reader->GetOutput(), (unsigned char)(cip::UPPERTHIRD) );
    
    if ( bbRegion.GetIndex()[0] != 0 || bbRegion.GetIndex()[1] != 0 || bbRegion.GetIndex()[2] != 0 ||
	 bbRegion.GetSize()[0] != 0 || bbRegion.GetSize()[1] != 0 || bbRegion.GetSize()[2] != 0 )
      {
	std::cout << "FAILED" << std::endl;
	return 1;
      }
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
