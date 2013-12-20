#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"
#include "itkCIPPartialLungLabelMapImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "cipConventions.h"
#include "GeneratePartialLungLabelMapCLP.h"

typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::ImageRegionIteratorWithIndex< UShortImageType >      UShortIteratorType;
typedef itk::CIPPartialLungLabelMapImageFilter< ShortImageType >  PartialLungType;
typedef itk::GDCMImageIO                                          ImageIOType;
typedef itk::GDCMSeriesFileNames                                  NamesGeneratorType;
typedef itk::ImageSeriesReader< ShortImageType >                  SeriesReaderType;
typedef itk::MedianImageFilter< ShortImageType, ShortImageType > MedianType;

void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );
ShortImageType::Pointer ReadCTFromDirectory( std::string );
ShortImageType::Pointer ReadCTFromFile( std::string );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  //
  // Read the CT image
  //
  ShortImageType::Pointer ctImage = ShortImageType::New();

  if ( ctDir.compare("NA") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = ReadCTFromDirectory( ctDir );
    }
  else if ( ctFileName.compare("NA") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
    ctImage = ReadCTFromFile( ctFileName );
    }
  else
    {
    std::cerr << "ERROR: No CT image specified" << std::endl;
    
    return 0;
    }

  ShortImageType::SpacingType spacing = ctImage->GetSpacing();
  
  unsigned long closingNeighborhood[3];
    closingNeighborhood[0] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[0] ) );
    closingNeighborhood[1] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[1] ) );
    closingNeighborhood[2] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[2] ) );

  closingNeighborhood[0] = closingNeighborhood[0]>0 ? closingNeighborhood[0] : 1;
  closingNeighborhood[1] = closingNeighborhood[1]>0 ? closingNeighborhood[1] : 1;
  closingNeighborhood[2] = closingNeighborhood[2]>0 ? closingNeighborhood[2] : 1;

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctImage, lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping low CT image values (enforcing -1024 lower bound)..." << std::endl;
  LowerClipImage( ctImage, -1024, -1024 );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctImage, upperClipValue, upperReplacementValue );

  {
  ShortImageType::SizeType medianRadius;
    medianRadius[0] = 1;
    medianRadius[1] = 1;
    medianRadius[2] = 1;

  std::cout << "Executing median filter..." << std::endl;
  MedianType::Pointer median = MedianType::New();
    median->SetInput( ctImage );
    median->SetRadius( medianRadius );
    median->Update();

  ShortIteratorType gIt( ctImage, ctImage->GetBufferedRegion() );
  ShortIteratorType mIt( median->GetOutput(), median->GetOutput()->GetBufferedRegion() );

  gIt.GoToBegin();
  mIt.GoToBegin();
  while ( !gIt.IsAtEnd() )
    {
    gIt.Set( mIt.Get() );

    ++gIt;
    ++mIt;
    }
  }

  std::cout << "Executing partial lung filter..." << std::endl;
  PartialLungType::Pointer partialLungFilter = PartialLungType::New();
    partialLungFilter->SetInput( ctImage );
  if ( aggressiveLungSplitting == 1 )
    {
    partialLungFilter->SetAggressiveLeftRightSplitter( true );
    }
  if ( headFirst == 1 )
    {
    partialLungFilter->SetHeadFirst( true );
    }
  else
    {
    partialLungFilter->SetHeadFirst( false );
    }
    partialLungFilter->SetMaxAirwayVolumeIncreaseRate( airwayVolumeIncreaseRate );
    partialLungFilter->SetLeftRightLungSplitRadius( lungSplitRadius );
    partialLungFilter->SetMinAirwayVolume( minAirwayVolume );
    partialLungFilter->SetMaxAirwayVolume( maxAirwayVolume );
    partialLungFilter->SetClosingNeighborhood( closingNeighborhood );
  //
  // Read the helper mask if specified
  //
  if ( helperMaskFileName.compare("NA") != 0 )
    {
    std::cout << "Reading helper mask..." << std::endl;
    UShortReaderType::Pointer helperReader = UShortReaderType::New();
      helperReader->SetFileName( helperMaskFileName );
    try
      {
      helperReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading helper mask:";
      std::cerr << excp << std::endl;
      }
    partialLungFilter->SetHelperMask( helperReader->GetOutput() );
    }
    partialLungFilter->Update();

  // Before writing, label by thirds. Here we are only concerned about upper, middle, 
  // and lower, regardless of left right.
  cip::ChestConventions conventions;
  unsigned int totVoxels = 0;

  UShortIteratorType lIt( partialLungFilter->GetOutput(), partialLungFilter->GetOutput()->GetBufferedRegion() );

  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  if ( conventions.GetChestRegionFromValue( lIt.Get() ) > 0 )
	    {
	      totVoxels++;
	    }
	}

      ++lIt;
    }

  unsigned int inc = 0;

  lIt.GoToBegin();
  while ( !lIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 )
	{
	  unsigned char cipRegion = conventions.GetChestRegionFromValue( lIt.Get() );
	  unsigned char cipType   = conventions.GetChestTypeFromValue( lIt.Get() );
	  if ( cipRegion > 0 )
	    {
	      inc++;

	      if ( double(inc)/double(totVoxels) < 1.0/3.0 )
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::LOWERTHIRD), cipType ) );
		}
	      else if ( double(inc)/double(totVoxels) < 2.0/3.0 )
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::MIDDLETHIRD), cipType ) );
		}
	      else 
		{
		  lIt.Set( conventions.GetValueFromChestRegionAndType( (unsigned char)(cip::UPPERTHIRD), cipType ) );
		}
	    }
	}      

      ++lIt;
    }

  std::cout << "Writing lung mask image..." << std::endl;
  UShortWriterType::Pointer maskWriter = UShortWriterType::New(); 
    maskWriter->SetInput( partialLungFilter->GetOutput() );
    maskWriter->SetFileName( outputLungMaskFileName );
    maskWriter->UseCompressionOn();
  try
    {
    maskWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing lung mask:";
    std::cerr << excp << std::endl;
    }

  std::cout << "DONE." << std::endl;

  return 0;
}


void LowerClipImage( ShortImageType::Pointer image, short clipValue, short replacementValue )
{
  ShortIteratorType iIt( image, image->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() < clipValue )
      {
      iIt.Set( replacementValue );
      }

    ++iIt;
    }
}


void UpperClipImage( ShortImageType::Pointer image, short clipValue, short replacementValue )
{
  ShortIteratorType iIt( image, image->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() > clipValue )
      {
      iIt.Set( replacementValue );
      }

    ++iIt;
    }
}


ShortImageType::Pointer ReadCTFromDirectory( std::string ctDir )
{
  ImageIOType::Pointer gdcmIO = ImageIOType::New();

  std::cout << "---Getting file names..." << std::endl;
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( ctDir );

  const SeriesReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

  std::cout << "---Reading DICOM image..." << std::endl;
  SeriesReaderType::Pointer dicomReader = SeriesReaderType::New();
    dicomReader->SetImageIO( gdcmIO );
    dicomReader->SetFileNames( filenames );
  try
    {
    dicomReader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Exception caught while reading dicom:";
    std::cerr << excp << std::endl;
    }  

  return dicomReader->GetOutput();
}


ShortImageType::Pointer ReadCTFromFile( std::string fileName )
{
  ShortReaderType::Pointer reader = ShortReaderType::New();
    reader->SetFileName( fileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;
    }

  return reader->GetOutput();
}
