/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program takes as an argument a directorying
 *  containing DICOM images, and produces a single file as
 *  output. Single files  are preferred for our operations as they
 *  compactly contain the CT data. 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "itkRGBPixel.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "ConvertDicomCLP.h"

typedef itk::Image< short, 3 >                ImageType;
typedef itk::GDCMImageIO                      ImageIOType;
typedef itk::GDCMSeriesFileNames              NamesGeneratorType;
typedef itk::ImageFileReader< ImageType >     ReaderHeaderType;
typedef itk::ImageSeriesReader< ImageType >   ReaderType;
typedef itk::ImageFileWriter< ImageType >     WriterType;
typedef itk::RGBPixel< unsigned char >       RGBPixelType;
typedef itk::Image< RGBPixelType, 3 >            RGBImageType;
typedef itk::ImageSeriesReader< RGBImageType >   RGBReaderType;
typedef itk::ImageFileWriter< RGBImageType >     RGBWriterType;
typedef itk::Image< float, 3 >                PETImageType;
typedef itk::ImageFileReader< PETImageType >     PETReaderHeaderType;
typedef itk::ImageSeriesReader< PETImageType >   PETReaderType;
typedef itk::ImageFileWriter< PETImageType >     PETWriterType;

int DoItCT( std::string, std::string );
int DoItUS( std::string, std::string );
int DoItPT( std::string, std::string );
std::string FindDicomTag( const std::string & , const ImageIOType::Pointer );

int main( int argc, char *argv[] )
{    
  PARSE_ARGS;  

    
  // Extracting header information to select proper reader
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
  namesGenerator->SetInputDirectory( dicomDir );
  namesGenerator->Update();

  // Set dicomIO and filenames
  ImageIOType::Pointer gdcmIO = ImageIOType::New();
  
  std::cout << "Getting file names..." << std::endl;
  std::vector< std::string > filenames = namesGenerator->GetInputFileNames();
    
  // Set up dummy reader
  std::cout << "Reading DICOM header information..." << std::endl;
  ReaderHeaderType::Pointer dicomReader = ReaderHeaderType::New();
  dicomReader->SetImageIO( gdcmIO );
  dicomReader->SetFileName( filenames[0] );
  try
    {
        dicomReader->Update();
    }
    catch (itk::ExceptionObject &excp)
    {
        std::cerr << "Exception caught while reading dicom header information:";
        std::cerr << excp << std::endl;
        return cip::DICOMREADFAILURE;
    }
    
  std::string modality  = FindDicomTag("0008|0060", gdcmIO);
    
  std::cout<<"Modality: "<<modality<<std::endl;

  int code;
  if (modality == "CT") {
      code=DoItCT( dicomDir, outputImageFileName);
    }
  else if (modality == "US")
    {
      code=DoItUS( dicomDir, outputImageFileName);
    }
  else if (modality == "PT")
    {
      code=DoItPT( dicomDir, outputImageFileName);
    }
  else
    {
      code=DoItCT( dicomDir, outputImageFileName);
    }
    
  std::cout << "DONE." << std::endl;

  return code;
}

std::string FindDicomTag( const std::string & entryId, const itk::GDCMImageIO::Pointer dicomIO )
{
    std::string tagvalue;
    bool found = dicomIO->GetValueFromTag(entryId, tagvalue);
    if ( !found )
    {
        tagvalue = "NOT FOUND";
    }
    return tagvalue;
}


int DoItCT( std::string dicomDir, std::string outputImageFileName )
{  
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( dicomDir );
    namesGenerator->Update();

  // Read the DICOM data
  ImageIOType::Pointer gdcmIO = ImageIOType::New();

  std::cout << "Getting file names..." << std::endl;
  std::vector< std::string > filenames = namesGenerator->GetInputFileNames();  
 
    
  // Write the DICOM data
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
    std::cerr << "Exception caught writing image:";
    std::cerr << excp << std::endl;
    return cip::NRRDWRITEFAILURE;
    }

  return cip::EXITSUCCESS;
}

int DoItPT( std::string dicomDir, std::string outputImageFileName )
{
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( dicomDir );
    namesGenerator->Update();

  // Read the DICOM data
  ImageIOType::Pointer gdcmIO = ImageIOType::New();

  std::cout << "Getting file names..." << std::endl;
  std::vector< std::string > filenames = namesGenerator->GetInputFileNames();
 
    
  // Write the DICOM data
  std::cout << "Reading DICOM image..." << std::endl;
  PETReaderType::Pointer dicomReader = PETReaderType::New();
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
  PETWriterType::Pointer writer = PETWriterType::New();
    writer->SetInput( dicomReader->GetOutput() );
    writer->UseCompressionOn();
    writer->SetFileName( outputImageFileName );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing image:";
    std::cerr << excp << std::endl;
    return cip::NRRDWRITEFAILURE;
    }

  return cip::EXITSUCCESS;
}


int DoItUS( std::string dicomDir, std::string outputImageFileName )
{
    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory( dicomDir );
    namesGenerator->Update();
    
    // Read the DICOM data
    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    
    std::cout << "Getting file names..." << std::endl;
    std::vector< std::string > filenames = namesGenerator->GetInputFileNames();
    

    std::cout << "Reading DICOM image..." << std::endl;
    RGBReaderType::Pointer dicomReader = RGBReaderType::New();
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
    RGBWriterType::Pointer writer = RGBWriterType::New();
    writer->SetInput( dicomReader->GetOutput() );
    writer->UseCompressionOn();
    writer->SetFileName( outputImageFileName );
    try
    {
        writer->Update();
    }
    catch ( itk::ExceptionObject &excp )
    {
        std::cerr << "Exception caught writing image";
        std::cerr << excp << std::endl;
        return cip::NRRDWRITEFAILURE;
    }
  
    return cip::EXITSUCCESS;
}



#endif
