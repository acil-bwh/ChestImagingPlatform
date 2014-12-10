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

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMatrix.h"
#include "itkTransformFileReader.h"
#include "itkResampleImageFilter.h"
//#include "itkMetaImageIO.h" // not needed (fix build error)
#include "ResampleCT3DCLP.h"
#include <itkCompositeTransform.h>
#include "itkImageRegistrationMethod.h"

namespace
{
typedef itk::Image< short, 3 >                                                  ShortImageType;
typedef itk::LinearInterpolateImageFunction< ShortImageType, double >  InterpolatorType;
typedef itk::ResampleImageFilter< ShortImageType,ShortImageType >               ResampleType;
typedef itk::AffineTransform< double, 3 >                                       TransformType;
typedef itk::CompositeTransform< double, 3 >                                    CompositeTransformType;
typedef itk::ImageFileWriter< ShortImageType >                                  ShortWriterType2D;
typedef itk::ImageFileReader< ShortImageType >                                  ShortReaderType;
typedef itk::ImageRegistrationMethod<ShortImageType,ShortImageType >            CTRegistrationType;

//TransformType::Pointer GetTransformFromFile( std::string );

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

 // transform->GetInverse( transform ); //Not sure about what this is doing here

  return transform;
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


int main( int argc, char *argv[] )
{

	PARSE_ARGS;
  
  //
  // Read the destination image information for spacing, origin, and
  // size information (neede for the resampling process).
  //
  ShortImageType::SpacingType spacing;
  ShortImageType::SizeType    size;
  ShortImageType::PointType   origin;
  
  std::cout << "Reading destination information..." << std::endl;
  ShortReaderType::Pointer destinationReader = ShortReaderType::New();
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
  // Read the ct image
  //
  std::cout << "Reading ct image..." << std::endl;
  ShortReaderType::Pointer shortReader = ShortReaderType::New();
    shortReader->SetFileName( labelMapFileName.c_str() );
  try
    {
    shortReader->Update();
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
    
    //last transform applied first, so make last transform
    CompositeTransformType::Pointer transform = CompositeTransformType::New();
    TransformType::Pointer transformTemp2 = TransformType::New();
    for ( unsigned int i=0; i<transformFileName.size(); i++ )
    { std::cout<<"adding tx: "<<i<<std::endl;
        TransformType::Pointer transformTemp = TransformType::New();
        transformTemp = GetTransformFromFile((transformFileName[i]).c_str() );
        // Invert the transformation if specified by command like argument. Only inverting the first transformation

        if((i==0)&& (isInvertTransformation == true))
        {
            std::cout<<"inverting transform"<<std::endl;
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
    resampler->SetInput( shortReader->GetOutput() );
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

  ShortWriterType2D::Pointer writer = ShortWriterType2D::New();
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
