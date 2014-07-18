/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program takes as an argument a directorying
 *  containing DICOM images, and produces a single file as
 *  output. Single files  are preferred for our operations as they
 *  compactly contain the CT data. 
 *
 *  USAGE: 
 *
 *  ConvertDicom  -o \<string\> -i \<string\> [--] [--version] [-h]
 *
 *  Where: 
 *
 *  -o \<string\>,  --output \<string\>
 *    (required)  Output image file name
 *
 *  -i \<string\>,  --dicomDir \<string\>
 *    (required)  Input dicom directory
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
 *  $Date: 2012-10-23 10:16:06 -0400 (Tue, 23 Oct 2012) $
 *  $Revision: 300 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "ConvertDicomCLP.h"

typedef itk::Image< short, 3 >                ImageType;
typedef itk::GDCMImageIO                      ImageIOType;
typedef itk::GDCMSeriesFileNames              NamesGeneratorType;
typedef itk::ImageSeriesReader< ImageType >   ReaderType;
typedef itk::ImageFileWriter< ImageType >     WriterType;

int main( int argc, char *argv[] )
{
    
  PARSE_ARGS;  

  //
  // Read the DICOM data
  //
  ImageIOType::Pointer gdcmIO = ImageIOType::New();

  std::cout << "Getting file names..." << std::endl;
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( dicomDir );

  const ReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

  //
  // Write the DICOM data
  //
  std::cout << "Reading DICOM image..." << std::endl;
  ReaderType::Pointer dicomReader = ReaderType::New();
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
    return cip::DICOMREADFAILURE;
    }
  
  std::cout << "Writing converted image..." << std::endl;
  WriterType::Pointer writer = WriterType::New();  
    writer->SetInput( dicomReader->GetOutput() );
    writer->UseCompressionOn();
    writer->SetFileName( outputImageFileName );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing imag:";
    std::cerr << excp << std::endl;
    return cip::NRRDWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
