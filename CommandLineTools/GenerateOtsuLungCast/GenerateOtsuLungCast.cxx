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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCIPOtsuLungCastImageFilter.h"

typedef itk::Image< unsigned short, 3 >                           UShortImageType;
typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< UShortImageType >                   UShortReaderType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< UShortImageType >                   UShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::CIPOtsuLungCastImageFilter< ShortImageType >         CIPOtsuCastType;


void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string  lungMaskFileName;
  std::string  ctFileName;
  short        lowerClipValue         = -1024;
  short        lowerReplacementValue  = -1024;
  short        upperClipValue         = 1024;
  short        upperReplacementValue  = 1024;

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( "This program produces an Otsu lung cast. (Cast is meant\
                        to refer to it being a preliminary mask from which other\
                        masks are derived / molded). The program simply interfaces\
                        with the itkCIPOtsuLungCastImageFilter. Before invoking the\
                        filter, however, the user has the option to clip the intensity\
                        values of the input image. It's generally recommend to clip\
                        anything below -1024 or above 1024 to 1024.",  
                       ' ', 
                       "$Revision: 93 $" );

    TCLAP::ValueArg< std::string > lungMaskFileNameArg ( "o", "mask", "Output lung mask file name", true, lungMaskFileName, "string", cl );
    TCLAP::ValueArg< std::string > ctFileNameArg ( "i", "ct", "Input CT image file name", true, ctFileName, "string", cl );
    TCLAP::ValueArg< short >       lowerClipValueArg ( "l", "lowerClip", "Lower clip value applied to input image before segmentation. Any value below the value specified with this flag will\
 be replaced with the value specified by the -r flag. If the -r flag is not used, a default value of -1024 will be used as the replacement \
 value.", false, lowerClipValue, "short", cl );
    TCLAP::ValueArg< short >       lowerReplacementValueArg ( "r", "lowerReplace", "Lower replacement value applied to input image before segmentation. The value specified with this flag will replace any\
 value below the value specified using the -l flag. If no value is specified with the -l flag, the default of -1024 will be used.", false, lowerReplacementValue, "short", cl );
    TCLAP::ValueArg< short >       upperClipValueArg ( "u", "upperClip", "Upper clip value applied to input image before segmentation.Any value above the value specified with this flag will be\
 replaced with the value specified by the -R flag. If the -R flag is not used, a default value of 1024 will be used as the replacement value.", false, upperClipValue, "short", cl );
    TCLAP::ValueArg< short >       upperReplacementValueArg ( "R", "upperReplace", "Upper replacement value applied to input image before segmentation. Any value above the value specified with this flag will\
 replace the value specified using the -u flag. If no value is specified with the -u flag, the default of 1024 will be used.", false, upperReplacementValue, "short", cl );

    cl.parse( argc, argv );

    lungMaskFileName      = lungMaskFileNameArg.getValue();
    ctFileName            = ctFileNameArg.getValue();
    lowerClipValue        = lowerClipValueArg.getValue();
    lowerReplacementValue = lowerReplacementValueArg.getValue();
    upperClipValue        = upperClipValueArg.getValue();
    upperReplacementValue = upperReplacementValueArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

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

#endif
