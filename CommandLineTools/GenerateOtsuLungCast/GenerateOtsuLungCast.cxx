/** \file
 *  \ingroup commandLineTools 
 *  \details This program produces an Otsu lung cast. (Cast is meant
 *  to refer to it being a preliminary mask from which other masks are
 *  derived / molded). The program simply interfaces with the
 *  itkCIPOtsuLungCastImageFilter. Before invoking the filter,
 *  however, the user has the option to clip the intensity values of
 *  the input image. It's generally recommend to clip anything below
 *  -1024 or above 1024 to 1024.
 * 
 * USAGE: 
 *
 *   ./GenerateOtsuLungCast  [-R \<short\>] [-u \<short\>] [-r \<short\>] [-l
 *                           \<short\>] -i \<string\> -o \<string\> [--]
 *                           [--version] [-h]
 *
 * Where: 
 *
 *   -R \<short\>,  --upperReplace \<short\>
 *     Upper replacement value applied to input image before segmentation.
 *     Any value above the value specified with this flag will replace the
 *     value specified using the -u flag. If no value is specified with the
 *     -u flag, the default of 1024 will be used.
 *
 *   -u \<short\>,  --upperClip \<short\>
 *     Upper clip value applied to input image before segmentation.Any value
 *     above the value specified with this flag will be replaced with the
 *     value specified by the -R flag. If the -R flag is not used, a default
 *     value of 1024 will be used as the replacement value.
 *
 *   -r \<short\>,  --lowerReplace \<short\>
 *     Lower replacement value applied to input image before segmentation.
 *     The value specified with this flag will replace any value below the
 *     value specified using the -l flag. If no value is specified with the
 *     -l flag, the default of -1024 will be used.
 *
 *   -l \<short\>,  --lowerClip \<short\>
 *     Lower clip value applied to input image before segmentation. Any value
 *     below the value specified with this flag will be replaced with the
 *     value specified by the -r flag. If the -r flag is not used, a default
 *     value of -1024 will be used as the replacement  value.
 *
 *   -i \<string\>,  --ct \<string\>
 *     (required)  Input CT image file name
 *
 *   -o \<string\>,  --mask \<string\>
 *     (required)  Output lung mask file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCIPOtsuLungCastImageFilter.h"
#include "GenerateOtsuLungCastCLP.h"
namespace
{
typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::CIPOtsuLungCastImageFilter< ShortImageType >         CIPOtsuCastType;


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

}
int main( int argc, char *argv[] )
{

  //
  // Parse the input arguments
  //
    PARSE_ARGS;

    short lowerClipValue        = (short)lowerClipValueTemp;
    short lowerReplacementValue = (short)lowerReplacementValueTemp;
    short upperClipValue        = (short) upperClipValueTemp;
    short upperReplacementValue = (short) upperReplacementValueTemp;
  //
  // Read the CT image
  //
  std::cout << "Reading CT..." << std::endl;
  ShortReaderType::Pointer ctReader = ShortReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctReader->GetOutput(), lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctReader->GetOutput(), upperClipValue, upperReplacementValue );

  std::cout << "Getting Otsu lung cast..." << std::endl;
  CIPOtsuCastType::Pointer castFilter = CIPOtsuCastType::New();
    castFilter->SetInput( ctReader->GetOutput() );
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
    }

  std::cout << "DONE." << std::endl;

  return 0;
}




