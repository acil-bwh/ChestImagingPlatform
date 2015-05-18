#include "cipHelper.cxx"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "GenerateMedianFilteredImageCLP.h"
#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMedianImageFilter.h"


int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  typedef itk::MedianImageFilter< cip::CTType, cip::CTType > MedianType;

  // Read the CT image
  cip::CTType::Pointer ctImage = cip::CTType::New();

  if ( strcmp( ctDir.c_str(), "NA") != 0 )
    {
    std::cout << "Reading CT from directory..." << std::endl;
    ctImage = cip::ReadCTFromDirectory( ctDir );
    if (ctImage.GetPointer() == NULL)
        {
        return cip::DICOMREADFAILURE;
        }
    }
  else if ( strcmp( ctFileName.c_str(), "NA") != 0 )
    {
    std::cout << "Reading CT from file..." << std::endl;
    ctImage = cip::ReadCTFromFile( ctFileName );
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

  cip::CTType::SizeType medianRadius;
    medianRadius[0] = radiusValue;
    medianRadius[1] = radiusValue;
    medianRadius[2] = radiusValue;

  std::cout << "Executing median filter..." << std::endl;
  MedianType::Pointer median = MedianType::New();
    median->SetInput( ctImage );
    median->SetRadius( medianRadius );
    median->Update();

  std::cout << "Writing filtered image..." << std::endl;
  cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
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
