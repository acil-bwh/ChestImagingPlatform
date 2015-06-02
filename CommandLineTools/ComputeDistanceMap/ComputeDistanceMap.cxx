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

  typedef itk::SignedMaurerDistanceMapImageFilter< cip::LabelMapType, cip::DistanceMapType >     SignedMaurerType;
  typedef itk::SignedDanielssonDistanceMapImageFilter< cip::LabelMapType, cip::DistanceMapType > SignedDanielssonType;
  typedef itk::ImageToImageFilter<cip::LabelMapType, cip::DistanceMapType > FilterType;
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
  if (downsampleFactor <= 1)
    {
      std::cout << "Skipping downsampling..." << std::endl;
      subSampledLabelMap = reader->GetOutput();
    } 
  else
    {
      std::cout << "Downsampling label map..." << std::endl;
      subSampledLabelMap = cip::DownsampleLabelMap( downsampleFactor, reader->GetOutput() );
    }
  
  FilterType::Pointer distanceMap;
  if (method == "Maurer")
  {
    SignedMaurerType::Pointer distanceMap_M = SignedMaurerType::New();
    distanceMap_M->SetInput( subSampledLabelMap );
    distanceMap_M->SetSquaredDistance( outputSquaredDistance );
    distanceMap_M->SetUseImageSpacing( true );
    distanceMap_M->SetInsideIsPositive( interiorIsPositive );
    //distanceMap_M->SetBackgroundValue(0);
    distanceMap = distanceMap_M;
  }
  else if (method == "Danielsson")
  {
    SignedDanielssonType::Pointer distanceMap_D = SignedDanielssonType::New();
    distanceMap_D->SetInput( subSampledLabelMap );
    distanceMap_D->SetUseImageSpacing( true );
    distanceMap_D->SetInsideIsPositive( interiorIsPositive );
    distanceMap_D->SetSquaredDistance( outputSquaredDistance );
    distanceMap = distanceMap_D;
    }
  else
    {
      std::cerr << "Not supported distance transform method"<<std::endl;
      
      return cip::EXITFAILURE;
      
    }
    
  
  std::cout << "Computing distance map..." << std::endl;

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

  cip::DistanceMapType::Pointer upSampledDistanceMap;
  if (downsampleFactor <= 1 )
    {
      upSampledDistanceMap = distanceMap->GetOutput();
    } 
  else
    {
      std::cout << "Upsampling distance map..." << std::endl;
      upSampledDistanceMap = cip::UpsampleDistanceMap( downsampleFactor, distanceMap->GetOutput() );
    }

  std::cout << "Writing to file..." << std::endl;
  cip::DistanceMapWriterType::Pointer writer = cip::DistanceMapWriterType::New();
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

