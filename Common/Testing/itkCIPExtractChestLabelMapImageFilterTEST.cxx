#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkImageRegionIterator.h"

int main( int argc, char* argv[] )
{
  typedef itk::CIPExtractChestLabelMapImageFilter< 3 >   ExtractorType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >  IteratorType;

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

  // First test: just extract the whole lung region
  {
    std::cout << "Extracting..." << std::endl;
    ExtractorType::Pointer extractor = ExtractorType::New();
      extractor->SetInput( reader->GetOutput() );
      extractor->SetChestRegion( 1 );
      extractor->Update();

    IteratorType eIt( extractor->GetOutput(), extractor->GetOutput()->GetBufferedRegion() );
    IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

    eIt.GoToBegin();
    rIt.GoToBegin();
    while ( !eIt.IsAtEnd() )
      {
	if ( (rIt.Get() == 2 || rIt.Get() == 3 || rIt.Get() == 771 || rIt.Get() == 515
	      || rIt.Get() == 770 || rIt.Get() == 514) && eIt.Get() != 1)
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }
	if ( !(rIt.Get() == 2 || rIt.Get() == 3 || rIt.Get() == 771 || rIt.Get() == 515
	       || rIt.Get() == 770 || rIt.Get() == 514) && eIt.Get() == 1)
	  {
	    std::cout << "FAILED" << std::endl;
	    return 1;
	  }
	if ( eIt.Get() != 0 && eIt.Get() != 1 )
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }

	++eIt;
	++rIt;
      }
  }

  // Second test: extract the airways
  {
    std::cout << "Extracting..." << std::endl;
    ExtractorType::Pointer extractor = ExtractorType::New();
      extractor->SetInput( reader->GetOutput() );
      extractor->SetChestType( 2 );
      extractor->Update();

    IteratorType eIt( extractor->GetOutput(), extractor->GetOutput()->GetBufferedRegion() );
    IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

    eIt.GoToBegin();
    rIt.GoToBegin();
    while ( !eIt.IsAtEnd() )
      {
	if ( (rIt.Get() == 512 || rIt.Get() == 514 || rIt.Get() == 515) && eIt.Get() != 512)
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }
	if ( !(rIt.Get() == 512 || rIt.Get() == 514 || rIt.Get() == 515) && eIt.Get() == 512)
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }
	if ( eIt.Get() != 0 && eIt.Get() != 512 )
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }

	++eIt;
	++rIt;
      }
  }

  // Third test: extract a region-type pair: RightLung, Vessel
  {
    std::cout << "Extracting..." << std::endl;
    ExtractorType::Pointer extractor = ExtractorType::New();
      extractor->SetInput( reader->GetOutput() );
      extractor->SetRegionAndType( 2, 3 );
      extractor->Update();

    IteratorType eIt( extractor->GetOutput(), extractor->GetOutput()->GetBufferedRegion() );
    IteratorType rIt( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

    eIt.GoToBegin();
    rIt.GoToBegin();
    while ( !eIt.IsAtEnd() )
      {
	if ( rIt.Get() == 770 && eIt.Get() != 770 )
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }
	if ( rIt.Get() != 770 && eIt.Get() == 770 )
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }
	if ( eIt.Get() != 0 && eIt.Get() != 770 )
	  {
	  std::cout << "FAILED" << std::endl;
	  return 1;
	  }

	++eIt;
	++rIt;
      }
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
