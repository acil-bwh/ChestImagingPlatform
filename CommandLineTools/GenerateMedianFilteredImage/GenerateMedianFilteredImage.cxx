#include "itkImageFileWriter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "GenerateMedianFilteredImageCLP.h"
#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{

typedef itk::Image< short, 3 >                                    ShortImageType;
typedef itk::ImageFileReader< ShortImageType >                    ShortReaderType;
typedef itk::ImageFileWriter< ShortImageType >                    ShortWriterType;
typedef itk::ImageRegionIteratorWithIndex< ShortImageType >       ShortIteratorType;
typedef itk::GDCMImageIO                                          ImageIOType;
typedef itk::GDCMSeriesFileNames                                  NamesGeneratorType;
typedef itk::ImageSeriesReader< ShortImageType >                  SeriesReaderType;

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

} // end of anonymous namespace

int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  typedef itk::MedianImageFilter< ShortImageType, ShortImageType > MedianType;

  short lowerClipValue = lowerClipValues[0];
  short lowerReplacementValue = lowerClipValues[1];
  short upperClipValue = upperClipValues[0];
  short upperReplacementValue = upperClipValues[1];
 
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

  //return cip::EXITSUCCESS;
  return EXIT_SUCCESS;
}
