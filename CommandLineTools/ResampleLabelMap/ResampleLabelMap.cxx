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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMatrix.h"
#include "itkTransformFileReader.h"
#include "itkResampleImageFilter.h"
#include "itkMetaImageIO.h"

typedef itk::Image< unsigned short, 3 >                                       LabelMapType;
typedef itk::ImageFileReader< LabelMapType >                                  ReaderType;
typedef itk::ImageFileWriter< LabelMapType >                                  WriterType;
typedef itk::NearestNeighborInterpolateImageFunction< LabelMapType, double >  InterpolatorType;
typedef itk::ResampleImageFilter< LabelMapType, LabelMapType >                ResampleType;
typedef itk::AffineTransform< double, 3 >                                     TransformType;

TransformType::Pointer GetTransformFromFile( std::string );

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string labelMapFileName;
  std::string transformFileName;
  std::string resampledFileName;
  std::string destinationFileName;

  std::string labelMapFileNameDescription  = "Label map file name to resample";
  std::string transformFileNameDescription = "Transform file name";
  std::string resampledFileNameDescription = "Resampled label map (output) file name";
  std::string destinationFileNameDescription = "Destinatin file name. This should be a header file that\
contains the necessary information (image spacing, origin, and size) for the resampling process";

  std::string programDescription = "This program resamples a label map using an affine transform (read from file)";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 93 $" );

    TCLAP::ValueArg< std::string > labelMapFileNameArg( "l", "labelmap", labelMapFileNameDescription, true, labelMapFileName, "string", cl );
    TCLAP::ValueArg< std::string > transformFileNameArg( "t", "transform", transformFileNameDescription, true, transformFileName, "string", cl );
    TCLAP::ValueArg< std::string > resampledFileNameArg( "r", "resample", resampledFileNameDescription, true, resampledFileName, "string", cl );
    TCLAP::ValueArg< std::string > destinationFileNameArg( "d", "destination", destinationFileNameDescription, true, destinationFileName, "string", cl );

    cl.parse( argc, argv );

    labelMapFileName    = labelMapFileNameArg.getValue();
    transformFileName   = transformFileNameArg.getValue();
    resampledFileName   = resampledFileNameArg.getValue();
    destinationFileName = destinationFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }
  
  //
  // Read the destination image information for spacing, origin, and
  // size information (neede for the resampling process).
  //
  LabelMapType::SpacingType spacing;
  LabelMapType::SizeType    size;
  LabelMapType::PointType   origin;
  {
  std::cout << "Reading destination information..." << std::endl;
  ReaderType::Pointer destinationReader = ReaderType::New();
    destinationReader->SetFileName( destinationFileName );
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
  size    = destinationReader->GetOutput()->GetBufferedRegion().GetSize();
  origin  = destinationReader->GetOutput()->GetOrigin();
  }

  //
  // Read the label map image
  //
  std::cout << "Reading label map image..." << std::endl;
  ReaderType::Pointer labelMapReader = ReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
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
  std::cout << "Reading label map image..." << std::endl;
  TransformType::Pointer transform = GetTransformFromFile( transformFileName );

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
  WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( resampledFileName );
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

  transform->GetInverse( transform );

  return transform;
}

#endif
