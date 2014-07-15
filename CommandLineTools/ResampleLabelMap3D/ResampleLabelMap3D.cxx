/** \file
 *  \ingroup commandLineTools 
 *  \details This program resamples a label map using an affine
 *  transform (read from file). 
 *
 * USAGE:
 *
 * ResampleLabelMap.exe  -d \<string\> -r \<string\> -t \<string\> -l
 *                       \<string\> [--] [--version] [-h]
 *
 * Where:
 *
 *   -d \<string\>,  --destination \<string\>
 *     (required)  Destinatin file name. This should be a header file
 *     thatcontains the necessary information (image spacing, origin, and
 *     size) for the resampling process
 *
 *   -r \<string\>,  --resample \<string\>
 *     (required)  Resampled label map (output) file name
 *
 *   -t \<string\>,  --transform \<string\>
 *     (required)  Transform file name
 *
 *   -l \<string\>,  --labelmap \<string\>
 *     (required)  Label map file name to resample
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
 */

#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkMatrix.h"
#include "itkTransformFileReader.h"
#include "itkResampleImageFilter.h"
//#include "itkMetaImageIO.h" // not needed (fix build error)
#include "ResampleLabelMap3DCLP.h"
#include <itkCompositeTransform.h>


namespace
{
  typedef itk::NearestNeighborInterpolateImageFunction< cip::LabelMapType, double >  InterpolatorType;
  typedef itk::ResampleImageFilter< cip::LabelMapType, cip::LabelMapType >           ResampleType;
  typedef itk::AffineTransform< double, 3 >                                          TransformType;
  typedef itk::CompositeTransform< double, 3 >                                   CompositeTransformType;

  TransformType::Pointer GetTransformFromFile( std::string fileName )
  {
    itk::TransformFileReader::Pointer transformReader = itk::TransformFileReader::New();
    transformReader->SetFileName( fileName );
    try
      {
	transformReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
	std::cerr << "Exception caught reading transform:";
	std::cerr << excp << std::endl;
      }
  
    itk::TransformFileReader::TransformListType::const_iterator it;
    it = transformReader->GetTransformList()->begin();
    TransformType::Pointer transform = static_cast< TransformType* >( (*it).GetPointer() ); 
    return transform;
  }

} // end of anonymous namespace


int main( int argc, char *argv[] )
{

  PARSE_ARGS;
  
  //
  // Read the destination image information for spacing, origin, and
  // size information (neede for the resampling process).
  //
  cip::LabelMapType::SpacingType spacing;
  cip::LabelMapType::SizeType    size;
  cip::LabelMapType::PointType   origin;
  
  std::cout << "Reading destination information..." << std::endl;
  cip::LabelMapReaderType::Pointer destinationReader = cip::LabelMapReaderType::New();
  destinationReader->SetFileName( destinationFileName.c_str() );
  try
    {
      destinationReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;

      return cip::LABELMAPREADFAILURE;
    }

  spacing = destinationReader->GetOutput()->GetSpacing();
  size    = destinationReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  origin  = destinationReader->GetOutput()->GetOrigin();
  

  //
  // Read the label map image
  //
  std::cout << "Reading label map image..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
  labelMapReader->SetFileName( labelMapFileName.c_str() );
  try
    {
      labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;

      return cip::LABELMAPREADFAILURE;
    }

  //
  // Read the transform
  //
      
  //last transform applied first
  CompositeTransformType::Pointer transform = CompositeTransformType::New();
  for ( unsigned int i=0; i<transformFileName.size(); i++ )
    { std::cout<<"adding tx: "<<i<<std::endl;
      TransformType::Pointer transformTemp = TransformType::New();
      transformTemp = GetTransformFromFile((transformFileName[i]).c_str() );
      // Invert the transformation if specified by command like argument.
      // Only inverting the first transformation

      if((i==0)&& (isInvertTransformation == true))
        {
	  transformTemp->GetInverse( transformTemp );
        }
      transform->AddTransform(transformTemp);
    } 
  transform->SetAllTransformsToOptimizeOn();		

  //
  // Resample the label map
  //
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  std::cout << "Resampling..." << std::endl;
  ResampleType::Pointer resampler = ResampleType::New();
  resampler->SetTransform( transform );
  resampler->SetInterpolator( interpolator );
  resampler->SetInput( labelMapReader->GetOutput() );
  resampler->SetSize( size );
  resampler->SetOutputSpacing( spacing );
  resampler->SetOutputOrigin( origin );
  resampler->SetOutputDirection( destinationReader->GetOutput()->GetDirection() );
  try
    {
      resampler->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught resampling:";
      std::cerr << excp << std::endl;

      return cip::RESAMPLEFAILURE;
    }

  //
  // Write the resampled label map to file
  //
  std::cout << "Writing resampled label map..." << std::endl;
  cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
  writer->SetFileName( resampledFileName.c_str());
  writer->UseCompressionOn();
  writer->SetInput( resampler->GetOutput() );
  try
    {
      writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught writing label map:";
      std::cerr << excp << std::endl;
    
      return cip::LABELMAPWRITEFAILURE;
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}
