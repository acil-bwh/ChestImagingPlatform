#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.h"
#include "itkImageRegionIterator.h"

int main( int argc, char* argv[] )
{
  typedef itk::CIPAutoThresholdAirwaySegmentationImageFilter< cip::CTType > AirwaySegmentationType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >                     IteratorType;

  // Read the CT image
  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
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

  cip::CTType::IndexType seed;
    seed[0] = atoi( argv[2] );
    seed[1] = atoi( argv[3] );
    seed[2] = atoi( argv[4] );

  std::cout << "Segmenting..." << std::endl;
  AirwaySegmentationType::Pointer segmenter = AirwaySegmentationType::New();
    segmenter->SetInput( reader->GetOutput() );
    segmenter->AddSeed( seed );
    segmenter->SetMinIntensityThreshold( -1024 );
    segmenter->SetMaxIntensityThreshold( -800 );
    segmenter->Update();

  if ( argc > 6 )
    {
      // Check the 6th param is not "EXTRAFILES" reserved word, 
      // because in this test it would mean more input files
      if (strncmp(argv[6], "EXTRAFILES", 10) != 0)
	{
	  std::cout << "Writing cast..." << std::endl;
	  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
	    writer->SetFileName( argv[6] );
	    writer->SetInput( segmenter->GetOutput() );
	    writer->UseCompressionOn();
	    writer->Update();
	}
    }
  
  // Read the reference label mape
  std::cout << "Reading reference..." << std::endl;
  cip::LabelMapReaderType::Pointer referenceReader = cip::LabelMapReaderType::New();
    referenceReader->SetFileName( argv[5] );
  try
    {
    referenceReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading reference label map:";
    std::cerr << excp << std::endl;
    }

  IteratorType segIt( segmenter->GetOutput(), segmenter->GetOutput()->GetBufferedRegion() );
  IteratorType refIt( referenceReader->GetOutput(), referenceReader->GetOutput()->GetBufferedRegion() );

  unsigned int refCount = 0;
  unsigned int segCount = 0;
  unsigned int intCount = 0;

  segIt.GoToBegin();
  refIt.GoToBegin();
  while ( !refIt.IsAtEnd() )
    {
      if ( refIt.Get() !=0 )
	{
	  refCount++;
	}
      if ( segIt.Get() != 0 )
	{
	  segCount++;
	}
      if ( refIt.Get() !=0 && segIt.Get() != 0 )
	{
	  intCount++;
	}

      ++segIt;
      ++refIt;
    }

  double dice = 2.0*double(intCount)/double(refCount + segCount);

  // The output segmentation with the above parameters was visually inspected
  // and seen to be reasonable. The resulting dice score was over 0.75.
  if ( dice > 0.75 )
    {
      std::cout << "PASSED" << std::endl;
      return 0;
    }

  std::cout << "FAILED" << std::endl;
  return 1;
}
