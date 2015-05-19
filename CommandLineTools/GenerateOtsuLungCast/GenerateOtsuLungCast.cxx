/** \file
 *  \ingroup commandLineTools 
 *  \details This program produces an Otsu lung cast. (Cast is meant
 *  to refer to it being a preliminary mask from which other masks are
 *  derived / molded). The program simply interfaces with the
 *  itkCIPOtsuLungCastImageFilter. Before invoking the filter,
 *  however, the user has the option to clip the intensity values of
 *  the input image. It's generally recommend to clip anything below
 *  -1024 or above 1024 to 1024.
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCIPOtsuLungCastImageFilter.h"
#include "GenerateOtsuLungCastCLP.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "cipHelper.h"

typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::CIPOtsuLungCastImageFilter< ShortImageType >         CIPOtsuCastType;

ShortImageType::Pointer ReadCTFromFile( std::string );
void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short  );

int main( int argc, char *argv[] )
{
  // Parse the input arguments
  PARSE_ARGS;

  short lowerClipValue        = (short)lowerClipValueTemp;
  short lowerReplacementValue = (short)lowerReplacementValueTemp;
  short upperClipValue        = (short)upperClipValueTemp;
  short upperReplacementValue = (short)upperReplacementValueTemp;

  // Read the CT image
  cip::CTType::Pointer ctImage = cip::CTType::New();

  if ( ctFileName.compare("NA") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
      ctImage = cip::ReadCTFromFile( ctFileName );
    }
  else
    {
    std::cerr << "ERROR: No CT image specified" << std::endl;
    
    return 0;
    }

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctImage, lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctImage, upperClipValue, upperReplacementValue );

  std::cout << "Getting Otsu lung cast..." << std::endl;
  CIPOtsuCastType::Pointer castFilter = CIPOtsuCastType::New();
    castFilter->SetInput( ctImage );
    castFilter->Update();

  std::cout << "Writing Otsu lung cast..." << std::endl;
  UShortWriterType::Pointer maskWriter = UShortWriterType::New(); 
    maskWriter->SetInput( castFilter->GetOutput() );
    maskWriter->SetFileName( lungMaskFileName );
    maskWriter->UseCompressionOn();
  try
    {
    maskWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing lung mask:";
    std::cerr << excp << std::endl;
    return cip::EXITFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
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
