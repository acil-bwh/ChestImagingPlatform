/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program takes as an argument a file patterns describing the
 *  content of a directory, and produces a single file as
 *  output. Single files  are preferred for our operations as they
 *  compactly contain the CT data. 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericSeriesFileNames.h"
#include "itkChangeInformationImageFilter.h"
#include "CreateVolumeFromSlicesCLP.h"

int main( int argc, char * argv[] )
{

  PARSE_ARGS;
  
  typedef short   PixelType;
  const unsigned int Dimension = 3;
  
  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::ImageSeriesReader< ImageType > ReaderType;
  typedef itk::ImageFileWriter<   ImageType > WriterType;
  typedef itk::ChangeInformationImageFilter< ImageType > FilterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  
  const unsigned int first = firstSliceValue;
  const unsigned int last  = lastSliceValue;
  
  typedef itk::NumericSeriesFileNames    NameGeneratorType;
  
  NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
  
  nameGenerator->SetSeriesFormat( filePattern.c_str() );
  
  nameGenerator->SetStartIndex( first );
  nameGenerator->SetEndIndex( last );
  nameGenerator->SetIncrementIndex( 1 );
  std::vector<std::string> names = nameGenerator->GetFileNames();
  
  // List the files
  //
  std::vector<std::string>::iterator nit;
  for (nit = names.begin();
       nit != names.end();
       nit++)
  {
    std::cout << "File: " << (*nit).c_str() << std::endl;
  }
  
  reader->SetFileNames( names  );
  
  
  
  // Set volume information (spacing and origin)
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput( reader->GetOutput() );
  
  ImageType::SpacingType sp_itk;
  ImageType::PointType or_itk;

  for (int ii =0; ii<Dimension; ii++){
    sp_itk[ii]=spacing[ii];
    or_itk[ii]=origin[ii];
  }
  filter->SetOutputSpacing( sp_itk );
  filter->ChangeSpacingOn();
  
  filter->SetOutputOrigin( or_itk );
  filter->ChangeOriginOn();
  
  filter->UpdateOutputInformation();
  
  writer->SetFileName( outputImageFileName.c_str() );
  writer->SetInput( filter->GetOutput() );
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return cip::NRRDWRITEFAILURE;
  }
  return cip::EXITSUCCESS;

}

#endif
