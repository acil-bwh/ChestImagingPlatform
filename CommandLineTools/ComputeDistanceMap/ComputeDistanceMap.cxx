#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSignedDanielssonDistanceMapImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "ComputeDistanceMapCLP.h"

namespace
{
  typedef itk::Image< short, 3 >                                                            DistanceMapType;
  typedef itk::ImageFileWriter< DistanceMapType >                                           WriterType;
  typedef itk::SignedMaurerDistanceMapImageFilter< cip::LabelMapType, DistanceMapType >     SignedMaurerType;
  typedef itk::SignedDanielssonDistanceMapImageFilter< cip::LabelMapType, DistanceMapType > SignedDanielssonType;
     
} // end of anonymous namespace


int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instantiate ChestConventions for convenience
  cip::ChestConventions conventions;

  // Read the label map image
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer reader = cip::LabelMapReaderType::New(); 
    reader->SetFileName( labelMapFileName );
  try
    {
    reader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading image:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  cip::LabelMapType::Pointer subSampledLabelMap;
  std::cout << "Downsampling label map..." << std::endl;  
  if (downsampleFactor >= 1)
    {
      subSampledLabelMap = reader->GetOutput();
    } 
  else
    {
      subSampledLabelMap = cip::DownsampleLabelMap( downsampleFactor, reader->GetOutput() );
    }
  
  std::cout << "Generating distance map..." << std::endl;
  SignedMaurerType::Pointer distanceMap = SignedMaurerType::New();
    distanceMap->SetInput( subSampledLabelMap );
    distanceMap->SetSquaredDistance( false );
    distanceMap->SetUseImageSpacing( true );
    distanceMap->SetInsideIsPositive( interiorIsPositive );
  try
    {
    distanceMap->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught generating distance map:";
    std::cerr << excp << std::endl;
    return cip::GENERATEDISTANCEMAPFAILURE;
    }

  DistanceMapType::Pointer upSampledDistanceMap;
  std::cout << "Upsampling distance map..." << std::endl;
  if (downsampleFactor >= 1 )
    {
      upSampledDistanceMap = distanceMap->GetOutput();
    } 
  else
    {
      upSampledDistanceMap = cip::UpsampleCT( downsampleFactor, distanceMap->GetOutput() );
    }

  std::cout << "Writing to file..." << std::endl;
  WriterType::Pointer writer = WriterType::New();
    writer->SetInput( upSampledDistanceMap );
    writer->SetFileName( distanceMapFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing image:";
    std::cerr << excp << std::endl;
    
    return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

