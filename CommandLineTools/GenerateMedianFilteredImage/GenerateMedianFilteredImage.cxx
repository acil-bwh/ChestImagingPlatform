/** \file
 *  \ingroup commandLineTools 
 *  \details This program can be used to median filter
 *   a CT image.

 *USAGE: 
 *
 *  ./GenerateMedianFilteredImage  [-r <unsigned int>] [--ucp <short>] ... 
 *                                 [--lcv <short>] ...  [-d <string>] -i
 *                                 <string> -o <string> [--] [--version]
 *                                 [-h]
 * Where: 
 *
 *  -r <unsigned int>,  --radius <unsigned int>
 *    Median filter radius (default =1)
 *
 *
 *  --ucp <short>  (accepted multiple times)
 *    Upper clip value applied to input image before filtering. This flag   
 *    should be followed by two values: the first value is the clip value
 *    and                the second value is the replacement value (i.e.,
 *    everything below the clip                value will be assigned the
 *    replacement value)
 *
 *  --lcv <short>  (accepted multiple times)
 *    Lower clip value applied to input image before filtering. This flag   
 *    should be followed by two values: the first value is the clip value
 *    and                the second value is the replacement value (i.e.,
 *    everything below the clip                value will be assigned the
 *    replacement value)
 *
 *  -d <string>,  --dir <string>
 *    Input CT directory
 *
 *  -i <string>,  --input <string>
 *    (required)  Input CT image file name
 *
 *  -o <string>,  --output <string>
 *    (required)  Output image file name
 *
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 *
 *  --version
 *    Displays version information and exits.
 *
 *  -h,  --help
 *    Displays usage information and exits.
 *
 *
 *  $Date: $
 *  $Revision: 317 $
 *  $Author: $
 *
 */
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"


typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< ShortImageType >                    ShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::GDCMImageIO                                          ImageIOType;
typedef itk::GDCMSeriesFileNames                                  NamesGeneratorType;
typedef itk::ImageSeriesReader< ShortImageType >                  SeriesReaderType;


void LowerClipImage( ShortImageType::Pointer, short, short );
void UpperClipImage( ShortImageType::Pointer, short, short );
ShortImageType::Pointer ReadCTFromDirectory( std::string );
ShortImageType::Pointer ReadCTFromFile( std::string  );



int main( int argc, char *argv[] )
{
  typedef itk::MedianImageFilter< ShortImageType, ShortImageType > MedianType;

  //
  // Define the arguments to be passed
  //
  std::string  outputFileName        = "q";
  std::string  ctFileName            = "q";
  std::string  ctDir                 = "q";
  short  lowerClipValue        = -1224;
  short  lowerReplacementValue = 1024;
  short  upperClipValue        = 1024;
  short  upperReplacementValue = 1024;
  unsigned int radiusValue     = 1;

  //
  // Argument description for user
  //
  std::string programDescription = "Median filter a CT volume";
  std::string outputFileNameDescription = "Output image file name";
  std::string ctFileNameDescription = "Input CT image file name";
  std::string ctDirDescription = "Input CT directory";
  std::string lowerClipValueDescription = "Lower clip value applied to input image before filtering. This flag \
               should be followed by two values: the first value is the clip value and \
               the second value is the replacement value (i.e., everything below the clip \
               value will be assigned the replacement value)\n";
  std::string upperClipValueDescription = "Upper clip value applied to input image before filtering. This flag \
               should be followed by two values: the first value is the clip value and \
               the second value is the replacement value (i.e., everything below the clip \
               value will be assigned the replacement value)\n";
  std::string radiusValueDescription = "Median filter radius (default =1)\n";

  //
  // Parse inputs
  //
  try
    {
     TCLAP::CmdLine cl (programDescription, ' ', "$Revision: 317 $");
     TCLAP::ValueArg<std::string> outputFileNameArg( "o", "output", outputFileNameDescription, true, outputFileName, "string", cl);
     TCLAP::ValueArg<std::string> ctFileNameArg ("i", "input", ctFileNameDescription, true, ctFileName, "string", cl);
     TCLAP::ValueArg<std::string> ctDirArg("d", "dir", ctDirDescription, false, ctDir, "string", cl);
     TCLAP::MultiArg<short> lowerClipValueArg ("","lcv", lowerClipValueDescription,false,"short", cl);
     TCLAP::MultiArg<short> upperClipValueArg ("","ucp", upperClipValueDescription,false,"short", cl);
     TCLAP::ValueArg<unsigned int> radiusValueArg ("r" ,"radius", radiusValueDescription,false,radiusValue, "unsigned int", cl);
     
     cl.parse( argc, argv);
 
     outputFileName = outputFileNameArg.getValue();     
     ctFileName     = ctFileNameArg.getValue();
     ctDir          = ctDirArg.getValue();
     radiusValue    = radiusValueArg.getValue();

     if (lowerClipValueArg.getValue().size() != 2) {
        TCLAP::ArgException("lcv is followed by two arguments");
     } else {
       lowerClipValue=lowerClipValueArg.getValue()[0];
       lowerReplacementValue = lowerClipValueArg.getValue()[1];
     }
     if (upperClipValueArg.getValue().size() != 2) {
        TCLAP::ArgException("lcv is followed by two arguments");
     } else {
       upperClipValue=upperClipValueArg.getValue()[0];
       upperReplacementValue = upperClipValueArg.getValue()[1];
     }
     
    }
  catch (TCLAP::ArgException excp)
    {
    std::cerr <<"Error: "<< excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }


  //-------
  // Read the CT image
  //
  ShortImageType::Pointer ctImage = ShortImageType::New();

  if ( strcmp( ctDir.c_str(), "q") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = ReadCTFromDirectory( ctDir );
    if (ctImage.GetPointer() == NULL)
        {
        return cip::DICOMREADFAILURE;
        }
    }
  else if ( strcmp( ctFileName.c_str(), "q") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
    ctImage = ReadCTFromFile( ctFileName );
    if (ctImage.GetPointer() == NULL)
        {
          return cip::NRRDREADFAILURE;
        }
    }
  else
    {
    std::cerr << "ERROR: No CT image specified" << std::endl;
    return cip::EXITFAILURE;
    }

  std::cout << "Clipping low CT image values..." << std::endl;
  LowerClipImage( ctImage, lowerClipValue, lowerReplacementValue );

  std::cout << "Clipping upper CT image values..." << std::endl;
  UpperClipImage( ctImage, upperClipValue, upperReplacementValue );

  ShortImageType::SizeType medianRadius;
    medianRadius[0] = radiusValue;
    medianRadius[1] = radiusValue;
    medianRadius[2] = radiusValue;

  std::cout << "Executing median filter..." << std::endl;
  MedianType::Pointer median = MedianType::New();
    median->SetInput( ctImage );
    median->SetRadius( medianRadius );
    median->Update();

  std::cout << "Writing filtered image..." << std::endl;
  ShortWriterType::Pointer writer = ShortWriterType::New(); 
    writer->SetInput( median->GetOutput() );
    writer->SetFileName( outputFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while writing filtered image:";
    std::cerr << excp << std::endl;
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
    return NULL;
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
    return NULL;
    }

  return reader->GetOutput();
}

#endif
