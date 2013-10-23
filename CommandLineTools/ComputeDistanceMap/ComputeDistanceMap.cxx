/** \file
 *  \ingroup commandLineTools 
 *  \details This program computes a distance map from an
 *  input binary map. A donwsampling can be applied prior to the distance map computation
 *  to improve performance. The resulting
 *  distance map will by upsampled by the same amount before writing.
 *  
 * USAGE:
 *
 * ComputeDistanceMap.exe  [-l <unsigned int>] 
 *                                      [-r <unsigned int>] 
 *                                      [-s <double>] -d <string> 
 *                                      -l <string> [--] [--version]
 *                                      [-h]
 *
 * Where:
 *
 *   -s <double>,  --downsample <double>
 *     Downsample factor. The input label map will be downsampled by the
 *     specified amount before the distance map is computed. The resulting
 *     distance map will then be scaled up by the same amount before
 *     writing.
 *
 *   -d <string>,  --distanceMap <string>
 *     (required)  Output distance map file name
 *
 *   -l <string>,  --labelMap <string>
 *     (required)  Input label map file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date: 2012-09-06 15:50:17 -0400 (Thu, 06 Sep 2012) $
 *  $Revision: 247 $
 *  $Author: rjosest $
 *
 */


#include "cipConventions.h"
#include "itkResampleImageFilter.h"
#include "itkImage.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkSignedDanielssonDistanceMapImageFilter.h"
#include "ComputeDistanceMapCLP.h"

namespace
{
typedef itk::Image< short, 3 >                                                            DistanceMapType;
typedef itk::ImageFileWriter< DistanceMapType >                                           WriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< cip::LabelMapType, DistanceMapType >     SignedMaurerType;
typedef itk::SignedDanielssonDistanceMapImageFilter< cip::LabelMapType, DistanceMapType > SignedDanielssonType;
typedef itk::NearestNeighborInterpolateImageFunction< cip::LabelMapType, double >         NearestNeighborInterpolatorType;
typedef itk::LinearInterpolateImageFunction< DistanceMapType, double >                    LinearInterpolatorType;
typedef itk::ResampleImageFilter< cip::LabelMapType, cip::LabelMapType >                  LabelMapResampleType;
typedef itk::ResampleImageFilter< DistanceMapType, DistanceMapType >                      DistanceMapResampleType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                            LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< DistanceMapType >                              DistanceMapIteratorType;
typedef itk::IdentityTransform< double, 3 >                                               IdentityType;


//cip::LabelMapType::Pointer ResampleImage( cip::LabelMapType::Pointer, float );
//DistanceMapType::Pointer ResampleImage( DistanceMapType::Pointer, cip::LabelMapType::SizeType, cip::LabelMapType::SpacingType );
    
    cip::LabelMapType::Pointer ResampleImage( cip::LabelMapType::Pointer image, float downsampleFactor )
    {
        cip::LabelMapType::SizeType inputSize = image->GetBufferedRegion().GetSize();
        
        cip::LabelMapType::SpacingType inputSpacing = image->GetSpacing();
        
        cip::LabelMapType::SpacingType outputSpacing;
        outputSpacing[0] = inputSpacing[0]*downsampleFactor;
        outputSpacing[1] = inputSpacing[1]*downsampleFactor;
        outputSpacing[2] = inputSpacing[2]*downsampleFactor;
        
        cip::LabelMapType::SizeType outputSize;
        outputSize[0] = static_cast< unsigned int >( static_cast< double >( inputSize[0] )/downsampleFactor );
        outputSize[1] = static_cast< unsigned int >( static_cast< double >( inputSize[1] )/downsampleFactor );
        outputSize[2] = static_cast< unsigned int >( static_cast< double >( inputSize[2] )/downsampleFactor );
        
        NearestNeighborInterpolatorType::Pointer interpolator = NearestNeighborInterpolatorType::New();
        
        IdentityType::Pointer transform = IdentityType::New();
        transform->SetIdentity();
        
        LabelMapResampleType::Pointer resampler = LabelMapResampleType::New();
        resampler->SetTransform( transform );
        resampler->SetInterpolator( interpolator );
        resampler->SetInput( image );
        resampler->SetSize( outputSize );
        resampler->SetOutputSpacing( outputSpacing );
        resampler->SetOutputOrigin( image->GetOrigin() );
        try
        {
            resampler->Update();
        }
        catch ( itk::ExceptionObject &excp )
        {
            std::cerr << "Exception caught down sampling:";
            std::cerr << excp << std::endl;
        }
        
        return resampler->GetOutput();
        
    }
    
    
    DistanceMapType::Pointer ResampleImage( DistanceMapType::Pointer image, cip::LabelMapType::SizeType outputSize, cip::LabelMapType::SpacingType outputSpacing )
    {
        //DistanceMapType::SizeType inputSize = image->GetBufferedRegion().GetSize();
        
        //DistanceMapType::SpacingType inputSpacing = image->GetSpacing();
        
        /*
         DistanceMapType::SpacingType outputSpacing;
         outputSpacing[0] = inputSpacing[0]*downsampleFactor;
         outputSpacing[1] = inputSpacing[1]*downsampleFactor;
         outputSpacing[2] = inputSpacing[2]*downsampleFactor;
         
         DistanceMapType::SizeType outputSize;
         outputSize[0] = static_cast< unsigned int >( static_cast< double >( inputSize[0] )/downsampleFactor );
         outputSize[1] = static_cast< unsigned int >( static_cast< double >( inputSize[1] )/downsampleFactor );
         outputSize[2] = static_cast< unsigned int >( static_cast< double >( inputSize[2] )/downsampleFactor );
         */
        
        LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
        
        IdentityType::Pointer transform = IdentityType::New();
        transform->SetIdentity();
        
        DistanceMapResampleType::Pointer resampler = DistanceMapResampleType::New();
        resampler->SetTransform( transform );
        resampler->SetInterpolator( interpolator );
        resampler->SetInput( image );
        resampler->SetSize( outputSize );
        resampler->SetOutputSpacing( outputSpacing );
        resampler->SetOutputOrigin( image->GetOrigin() );
        try
        {
            resampler->Update();
        }
        catch ( itk::ExceptionObject &excp )
        {
            std::cerr << "Exception caught down sampling:";
            std::cerr << excp << std::endl;
        }
        
        
        return resampler->GetOutput();
        
    } 

} // end of anonymous namespace


int main( int argc, char *argv[] )
{

    PARSE_ARGS;

  //
  // Instantiate ChestConventions for convenience
  //
  cip::ChestConventions conventions;

  //
  // Read the label map image
  //
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
  
  if (downsampleFactor == 1)
  {
    subSampledLabelMap = reader->GetOutput();

  } else
  {
     subSampledLabelMap=ResampleImage( reader->GetOutput(), downsampleFactor);
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
  if (downsampleFactor == 1 )
  {
    upSampledDistanceMap = distanceMap->GetOutput();
  } else
  {
    upSampledDistanceMap=ResampleImage( distanceMap->GetOutput(),
                                       reader->GetOutput()->GetBufferedRegion().GetSize(), reader->GetOutput()->GetSpacing() );
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

