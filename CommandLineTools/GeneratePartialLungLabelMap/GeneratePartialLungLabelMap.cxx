#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"
#include "itkCIPPartialLungLabelMapImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "GeneratePartialLungLabelMapCLP.h"

typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::ImageRegionIteratorWithIndex< UShortImageType >      UShortIteratorType;
typedef itk::CIPPartialLungLabelMapImageFilter< ShortImageType >  PartialLungType;

void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
    reader->SetFileName( ctFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( reader->GetOutput(), lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( reader->GetOutput(), upperClipValue, upperReplacementValue );

  std::cout << "Executing partial lung filter..." << std::endl;
  PartialLungType::Pointer partialLungFilter = PartialLungType::New();
    partialLungFilter->SetInput( reader->GetOutput() );
    partialLungFilter->SetAirwayMinIntensityThreshold( airwayMinThreshold );
    partialLungFilter->SetAirwayMaxIntensityThreshold( airwayMaxThreshold );
    partialLungFilter->SetMaxAirwayVolume( maxAirwayVolume );
  if ( feetFirst == true )
    {
    partialLungFilter->SetHeadFirst( false );
    }
  else
    {
    partialLungFilter->SetHeadFirst( true );
    }
    partialLungFilter->SetLeftRightLungSplitRadius( lungSplitRadius );
  try
    {
      partialLungFilter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught segmenting lungs:";
      std::cerr << excp << std::endl;

      return cip::SEGMENTATIONFAILURE;
    }

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
    maskWriter->SetFileName( outputLungMaskFileName.c_str() );
    maskWriter->UseCompressionOn();
  try
    {
    maskWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing lung mask:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPWRITEFAILURE;
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
