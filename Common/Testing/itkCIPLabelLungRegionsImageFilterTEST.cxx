#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImageRegionIterator.h"
#include "cipExceptionObject.h"
#include "itkCIPLabelLungRegionsImageFilter.h"

int main( int argc, char* argv[] )
{
  typedef itk::ImageRegionIterator< cip::LabelMapType >                                IteratorType;
  typedef itk::CIPLabelLungRegionsImageFilter< cip::LabelMapType, cip::LabelMapType >  LungRegionLabelerType;

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( argv[1] );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Labeling..." << std::endl;
  LungRegionLabelerType::Pointer labeler = LungRegionLabelerType::New();
    labeler->SetInput( labelMapReader->GetOutput() );
    labeler->LabelLeftAndRightLungsOn();
    labeler->SetHeadFirst( true );
    labeler->SetSupine( true );
    labeler->Update();

  // Read the reference label mape
  std::cout << "Reading reference..." << std::endl;
  cip::LabelMapReaderType::Pointer referenceReader = cip::LabelMapReaderType::New();
    referenceReader->SetFileName( argv[2] );
  try
    {
    referenceReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading reference label map:";
    std::cerr << excp << std::endl;
    }

  IteratorType lIt( labeler->GetOutput(), labeler->GetOutput()->GetBufferedRegion() );
  IteratorType rIt( referenceReader->GetOutput(), referenceReader->GetOutput()->GetBufferedRegion() );

  unsigned int refCount = 0;
  unsigned int segCount = 0;
  unsigned int intCount = 0;

  lIt.GoToBegin();
  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
      if ( rIt.Get() !=0 )
  	{
  	  refCount++;
  	}
      if ( lIt.Get() != 0 )
  	{
  	  segCount++;
  	}
      if ( rIt.Get() !=0 && lIt.Get() != 0 )
  	{
  	  intCount++;
  	}

      ++lIt;
      ++rIt;
    }

  double dice = 2.0*double(intCount)/double(refCount + segCount);

  // Changes to the splitter might result in tiny differences in the way
  // the split is performed. What we really care about is whether the
  // resulting split label map can be labeled as left-right
  if ( dice > 0.99 )
    {
      std::cout << "PASSED" << std::endl;
      return 0;
    }

  return 1;
}
