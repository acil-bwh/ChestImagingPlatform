#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkCIPSplitLeftLungRightLungImageFilter.h"
#include "itkImageRegionIterator.h"
#include "cipExceptionObject.h"
#include "itkCIPLabelLungRegionsImageFilter.h"

int main( int argc, char* argv[] )
{
  typedef itk::CIPSplitLeftLungRightLungImageFilter< cip::CTType >                     SplitterType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >                                IteratorType;
  typedef itk::CIPLabelLungRegionsImageFilter< cip::LabelMapType, cip::LabelMapType >  LungRegionLabelerType;

  // Read the CT image
  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( argv[1] );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Reading pre-split label map..." << std::endl;
  cip::LabelMapReaderType::Pointer preReader = cip::LabelMapReaderType::New();
    preReader->SetFileName( argv[2] );
  try
    {
    preReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading test image:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Splitting..." << std::endl;
  SplitterType::Pointer splitter = SplitterType::New();
    splitter->SetInput( ctReader->GetOutput() );
    splitter->SetLungLabelMap( preReader->GetOutput() );
  try
    {
      splitter->Update();
    }
  catch ( cip::ExceptionObject &excp )
    {
      std::cerr << "Exception caught splitting:";
      std::cerr << excp << std::endl;
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught splitting:";
      std::cerr << excp << std::endl;
    }

  std::cout << "Labeling..." << std::endl;
  LungRegionLabelerType::Pointer labeler = LungRegionLabelerType::New();
    labeler->SetInput( splitter->GetOutput() );
    labeler->LabelLeftAndRightLungsOn();
    labeler->SetHeadFirst( true );
    labeler->SetSupine( true );
    labeler->Update();

  // Read the reference label mape
  std::cout << "Reading reference..." << std::endl;
  cip::LabelMapReaderType::Pointer referenceReader = cip::LabelMapReaderType::New();
    referenceReader->SetFileName( argv[3] );
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

  std::cout << "FAILED" << std::endl;
  return 1;
}
