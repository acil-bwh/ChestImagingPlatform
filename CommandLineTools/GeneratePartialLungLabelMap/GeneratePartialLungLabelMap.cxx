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

void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );
ShortImageType::Pointer ReadCTFromDirectory( std::string );
ShortImageType::Pointer ReadCTFromFile( std::string );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Read the CT image
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

  // ShortImageType::SpacingType spacing = ctImage->GetSpacing();
  
  // unsigned long closingNeighborhood[3];
  //   closingNeighborhood[0] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[0] ) );
  //   closingNeighborhood[1] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[1] ) );
  //   closingNeighborhood[2] = static_cast< unsigned long >( vnl_math_rnd( closingRadius/spacing[2] ) );

  // closingNeighborhood[0] = closingNeighborhood[0]>0 ? closingNeighborhood[0] : 1;
  // closingNeighborhood[1] = closingNeighborhood[1]>0 ? closingNeighborhood[1] : 1;
  // closingNeighborhood[2] = closingNeighborhood[2]>0 ? closingNeighborhood[2] : 1;

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctImage, lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctImage, upperClipValue, upperReplacementValue );

  std::cout << "Executing partial lung filter..." << std::endl;
  PartialLungType::Pointer partialLungFilter = PartialLungType::New();
    partialLungFilter->SetInput( ctImage );
    partialLungFilter->SetAirwayMinIntensityThreshold( airwayMinThreshold );
    partialLungFilter->SetAirwayMaxIntensityThreshold( airwayMaxThreshold );
    partialLungFilter->SetLeftRightLungSplitRadius( leftRightLungSplitRadius );
  if ( feetFirst == true )
    {
    partialLungFilter->SetHeadFirst( false );
    }
  else
    {
    partialLungFilter->SetHeadFirst( true );
    }
    partialLungFilter->SetLeftRightLungSplitRadius( lungSplitRadius );
    partialLungFilter->Update();

  // //
  // // Read the helper mask if specified
  // //
  // if ( helperMaskFileName.compare("NA") != 0 )
  //   {
  //   std::cout << "Reading helper mask..." << std::endl;
  //   UShortReaderType::Pointer helperReader = UShortReaderType::New();
  //     helperReader->SetFileName( helperMaskFileName );
  //   try
  //     {
  //     helperReader->Update();
  //     }
  //   catch ( itk::ExceptionObject &excp )
  //     {
  //     std::cerr << "Exception caught reading helper mask:";
  //     std::cerr << excp << std::endl;
  //     }
  //   partialLungFilter->SetHelperMask( helperReader->GetOutput() );
  //   }
  //   partialLungFilter->Update();


  std::cout << "Writing lung mask image..." << std::endl;
  cip::LabelMapWriterType::Pointer maskWriter = cip::LabelMapWriterType::New(); 
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
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
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
