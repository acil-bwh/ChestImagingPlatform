/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a label map image and a chest region and type and
 *  generates a convex hull image corresponding it.
 *
 *  The algorithm proceeds by reading in the label appropriate to the right regio
 *  and type. That mask is then downsampled for faster  processing. After downsampling, 
 *  the convex hull is created. The convex hull is represented as a binary image (0 = background, 1 =
 *  foreground). The convex hull is upsampled so that it has the same
 *  extent as the original image, and it is then written to file.
 *
 *  USAGE:
 *
 *  GenerateSliceBasedConvexHull.exe [-s \<float\>]
 *                              [-d \<float\>] [-n \<int\>] 
 *                              [-o \<string\>] -r \<string\> -l \<string\>
 *                              [--] [--version] [-h]
 *
 * Where:
 *
 *
 *   -s \<float\>,  --sample \<float\>
 *     Down sample factor (default is 1)
 *
 *   -d \<float\>,  --degrees \<float\>
 *     Degrees resolution. This quanity relates to the accuracy of the
 *     finalconvex hull. Decreasing the degrees resolution increases
 *     accuracy. If this quantity changes, so should the number of rotations
 *     parameter(specified by the -nr flag). E.g. if number of rotations
 *     increases by a factor of two, degrees resolution should decrease by a
 *     factor of two(Default is 45.0 degrees)
 *
 *   -n \<int\>,  --numRotations \<int\>
 *     Number of rotations. This quanity relates to the accuracy of the
 *     finalconvex hull. Increasing the number of rotations increases
 *     accuracy. If this quantity changes, so should the resolution degrees
 *     parameter(specified by the -dr flag). E.g. if number of rotations
 *     increases by a factor of two, degrees resolution should decrease by a
 *     factor of two.
 *
 *   -o \<string\>,  --output \<string\>
 *     Output convex hull file name
 *
 *   -m \<string\>,  --mask \<string\>
 *     (required)  Input mask file name
 *
 *   -l \<int\>,  --regionAndTypeMaskLabel \<string\>
 *     (required)  Region and type index
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
 *  $Date: $
 *  $Revision:  $
 *  $Author: $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkIdentityTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkExtractImageFilter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkResampleImageFilter.h"


typedef itk::Image< unsigned short, 3 >                                     ImageType;
typedef itk::Image< unsigned short, 2 >                                     SliceType;
typedef itk::ImageFileReader< ImageType >                                   ReaderType;
typedef itk::ImageFileWriter< ImageType >                                   WriterType;
typedef itk::ImageRegionIterator< ImageType >                               IteratorType;
typedef itk::IdentityTransform< double, 3 >                                 IdentityType;
typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double >   ImageInterpolatorType;
typedef itk::NearestNeighborInterpolateImageFunction< SliceType, double >   SliceInterpolatorType;
typedef itk::ResampleImageFilter< ImageType, ImageType >                    ImageResampleType;
typedef itk::AffineTransform< double, 2 >                                   TransformType;
typedef itk::ResampleImageFilter< SliceType, SliceType >                    SliceResampleType;
typedef itk::ExtractImageFilter< ImageType, SliceType >                     SliceExtractorType;
typedef itk::ImageRegionIterator< SliceType >                               SliceIteratorType;  
typedef itk::ImageLinearIteratorWithIndex< SliceType >                      LinearIteratorType;


unsigned short GetMaxValueInImage( ImageType::Pointer );
void ReassignImageToConvexHull( ImageType::Pointer, int, float );
void ResampleImage( ImageType::Pointer, ImageType::Pointer, float );


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string maskFileName;
  std::string outputFileName;
  int         numRotations         = 1;
  float       degreesResolution    = 45.0;
  float       downsampleFactor     = 1.0;
  int         regionMaskLabel      = 0;

  //std::cout << "Chest Region:\t" << conventions.GetChestRegionNameFromValue( value ) << std::endl;
  //std::cout << "Chest Value:\t"  << conventions.GetChestTypeNameFromValue( value ) << std::endl;
  //cipConventions.h for how region and type are (unsigned char)
  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( "This program ....", 
                       ' ', 
                       "$Revision: 232 $" );

    TCLAP::ValueArg< std::string > maskFileNameArg ( "m", "maskFile", "Mask file name", true, maskFileName, "string", cl );
    TCLAP::ValueArg< std::string > outputFileNameArg ( "o", "output", "Output convex hull file name", false, outputFileName, "string", cl );
    TCLAP::ValueArg< int >         numRotationsArg ( "n", "numRotations", "Number of rotations. This quanity relates to the accuracy of the final\
convex hull. Increasing the number of rotations increases accuracy. If this quantity changes, so should the resolution degrees parameter\
(specified by the -dr flag). E.g. if number of rotations increases by a factor of two, degrees resolution should decrease by a factor of two.",
                                                     false, numRotations, "int", cl );
    TCLAP::ValueArg< float >       degreesResolutionArg ( "d", "degrees", "Degrees resolution. This quanity relates to the accuracy of the final\
convex hull. Decreasing the degrees resolution increases accuracy. If this quantity changes, so should the number of rotations parameter\
(specified by the -nr flag). E.g. if number of rotations increases by a factor of two, degrees resolution should decrease by a factor of two\
(Default is 45.0 degrees)", false, degreesResolution, "float", cl );
    TCLAP::ValueArg< float >       downsampleFactorArg ( "s", "sample", "Down sample factor (default is 1)", false, downsampleFactor, "float", cl );
       TCLAP::ValueArg< int >         regionMaskLabelArg ( "l", "regionMaskLabel", "The short value corresponding to the region and type for which the convex Hull should be computed.",
                                                     false, regionMaskLabel, "int", cl );
    cl.parse( argc, argv );

    maskFileName    = maskFileNameArg.getValue();
    outputFileName       = outputFileNameArg.getValue();
    numRotations         = numRotationsArg.getValue();
    degreesResolution    = degreesResolutionArg.getValue();
    downsampleFactor     = downsampleFactorArg.getValue();
	regionMaskLabel = regionMaskLabelArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  ImageType::Pointer completeThresholdedAtlas = ImageType::New();

  //
  // Read the label mask image. 
  //
  std::cout << "Reading mask image with label..." << regionMaskLabel<<std::endl;
  ReaderType::Pointer maskReader = ReaderType::New();
    maskReader->SetFileName( maskFileName );
  try
    {
    maskReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading left atlas:";
    std::cerr << excp << std::endl;

    return cip::ATLASREADFAILURE;
    }

  //Label all voxels in the mask image that have the desired label mask to 1, otherwise to 0

  completeThresholdedAtlas->SetRegions( maskReader->GetOutput()->GetBufferedRegion().GetSize() );
  completeThresholdedAtlas->Allocate();
  completeThresholdedAtlas->FillBuffer( 0 );
  completeThresholdedAtlas->SetSpacing( maskReader->GetOutput()->GetSpacing() );
  completeThresholdedAtlas->SetOrigin( maskReader->GetOutput()->GetOrigin() );

  IteratorType lIt( maskReader->GetOutput(), maskReader->GetOutput()->GetBufferedRegion() );
  IteratorType it( completeThresholdedAtlas, completeThresholdedAtlas->GetBufferedRegion() );

  lIt.GoToBegin();
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( lIt.Get() == short(regionMaskLabel) )
      {
      it.Set( 1 );
      }

    ++lIt;
    ++it;
    }

 
  //
  // Before computing the convex hull, subsample the image
  //
  ImageType::Pointer subSampledMask = ImageType::New();

  std::cout << "Subsampling atlas..." << std::endl;

  ResampleImage( completeThresholdedAtlas, subSampledMask, downsampleFactor );

  //
  // Now compute the convex hull
  //
  std::cout << "Computing convex hull..." << std::endl;
  ReassignImageToConvexHull( subSampledMask, numRotations, degreesResolution );  

  //
  // Up-sample the image
  //
  ImageType::Pointer convexHullImage = ImageType::New();

  std::cout << "Up-sampling atlas..." << std::endl;
  ResampleImage( subSampledMask, convexHullImage, 1.0/downsampleFactor );

  //
  // Write the convex hull
  //
  std::cout << "Writing convex hull..." << std::endl;
  WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputFileName );
    writer->UseCompressionOn();
    writer->SetInput( convexHullImage );
  try
    {
    writer->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing convex hull image:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPWRITEFAILURE;
    }
 
  return cip::EXITSUCCESS;
}


void ReassignImageToConvexHull( ImageType::Pointer image, int numRotations, float degreesResolution )
{
  ImageType::SizeType    imageSize    = image->GetBufferedRegion().GetSize();
  ImageType::SpacingType imageSpacing = image->GetSpacing();
  ImageType::PointType   imageOrigin  = image->GetOrigin();

  SliceType::SizeType sliceSize;
    sliceSize[0] = imageSize[0];
    sliceSize[1] = imageSize[1];

  ImageType::SizeType sliceExtractorSize;
    sliceExtractorSize[0] = imageSize[0];
    sliceExtractorSize[1] = imageSize[1];
    sliceExtractorSize[2] = 0;

  SliceType::SpacingType sliceSpacing;
    sliceSpacing[0] = imageSpacing[0];
    sliceSpacing[1] = imageSpacing[1];

  SliceType::PointType sliceOrigin;
    sliceOrigin[0] = imageOrigin[0];
    sliceOrigin[1] = imageOrigin[1];

  ImageType::IndexType sliceStartIndex;
    sliceStartIndex[0] = 0;
    sliceStartIndex[1] = 0;

  ImageType::RegionType sliceRegion;
    sliceRegion.SetSize( sliceExtractorSize );

  SliceInterpolatorType::Pointer interpolator = SliceInterpolatorType::New();

  TransformType::Pointer identity = TransformType::New();
    identity->SetIdentity();

  //
  // Set up the resample. The remainder of the necessary inputs will
  // be set during each slice's convex hull computation below
  //
  SliceResampleType::Pointer extractorResampler = SliceResampleType::New();
    extractorResampler->SetInterpolator( interpolator );
    extractorResampler->SetDefaultPixelValue( 0 );
    extractorResampler->SetOutputSpacing( sliceSpacing );
    extractorResampler->SetOutputOrigin( sliceOrigin );
    extractorResampler->SetSize( sliceSize );

  //
  // 'slice' will keep track of the filled in regions during the
  // convex hull computation (we'll reset it to zero before we begin
  // work on the next slice
  //
  SliceType::Pointer slice = SliceType::New();
    slice->SetRegions( sliceSize );
    slice->Allocate();
    slice->SetSpacing( sliceSpacing );
    slice->SetOrigin( sliceOrigin );

  //
  // translation1 and translation2 will be used to construct the
  // rotation transform so that the rotation occurs around the slice
  // center.
  //
  TransformType::OutputVectorType translation1;

  const double imageCenterX = imageOrigin[0] + imageSpacing[0] * imageSize[0] / 2.0;
  const double imageCenterY = imageOrigin[1] + imageSpacing[1] * imageSize[1] / 2.0;

  translation1[0] = -imageCenterX;
  translation1[1] = -imageCenterY;

  const double degreesToRadians = atan(1.0) / 45.0;
  double angle;

  TransformType::OutputVectorType translation2;
    translation2[0] =   imageCenterX;
    translation2[1] =   imageCenterY;

  for ( unsigned int i=0; i<imageSize[2]; i++ )
    {
    sliceStartIndex[2] = i;
    sliceRegion.SetIndex( sliceStartIndex );

    SliceExtractorType::Pointer sliceExtractor = SliceExtractorType::New();
      sliceExtractor->SetInput( image );
      sliceExtractor->SetExtractionRegion( sliceRegion );
      sliceExtractor->Update();

    extractorResampler->SetInput( sliceExtractor->GetOutput() );
    extractorResampler->SetTransform( identity );
    extractorResampler->Update();

    // 
    // For each extracted slice, initialize 'slice' to 0
    //
    slice->FillBuffer( 0 );

    //
    // First, check to see if there are any non-zero voxels in this
    // slice. We will do nothing if there are not
    //
    bool computeHull = false;

    SliceIteratorType erIt( extractorResampler->GetOutput(), extractorResampler->GetOutput()->GetBufferedRegion() );

    erIt.GoToBegin();
    while ( !erIt.IsAtEnd() )
      {
	if ( erIt.Get() == static_cast< unsigned short >( cip::WHOLELUNG ) )
	  {
	    computeHull = true;
	    
	    break;
	  }
	
	++erIt;
      }

    if ( computeHull )
      {
      for (int k=0; k<=numRotations; k++)
        {
        if ( k == 0 )
          {
          angle = 0.0;
          }
        else
          {
          angle = degreesResolution*degreesToRadians;
          }

        TransformType::Pointer transform = TransformType::New();
          transform->Translate( translation1 );
          transform->Rotate2D( angle, false );
          transform->Translate( translation2, false );

        SliceIteratorType sIt( slice, slice->GetBufferedRegion() );

        sIt.GoToBegin();
        erIt.GoToBegin();
        while ( !sIt.IsAtEnd() )
          {
          sIt.Set( erIt.Get() );

          ++sIt;
          ++erIt;
          }

        extractorResampler->SetInput( slice );
        extractorResampler->SetTransform( transform );
        extractorResampler->Update();

        //
        // Line Scan the resampled slice and fill (horizontal)
        //
        LinearIteratorType scanIt( extractorResampler->GetOutput(), extractorResampler->GetOutput()->GetBufferedRegion() );
        
        bool fromLeft, fromRight, fromTop, fromBottom;
        SliceType::IndexType leftIndex, rightIndex, topIndex, bottomIndex;

        bool continueToAdd = true;
        while ( continueToAdd )
          {
          continueToAdd = false;

          scanIt.SetDirection( 0 );
          for ( scanIt.GoToBegin(); !scanIt.IsAtEnd(); scanIt.NextLine() )
            {
            fromLeft  = false;
            fromRight = false;

            scanIt.GoToBeginOfLine();
            while ( !scanIt.IsAtEndOfLine() )
              {
	      if ( (scanIt.Get() == static_cast< unsigned short >( cip::WHOLELUNG ) ) && !fromLeft )
                {
                fromLeft = true;
                leftIndex = scanIt.GetIndex();        
                }

              ++scanIt;
              }

            //
            // At this point, the iterator can be past the buffer, so
            // scoot it back one spot so our first evaluation of the
            // label map value is actually valid
            //
            --scanIt;
            while ( !scanIt.IsAtReverseEndOfLine() )
              {
	      if ( (scanIt.Get() == static_cast< unsigned short >( cip::WHOLELUNG )) && !fromRight )
                {
                fromRight = true;
                rightIndex = scanIt.GetIndex();
                }
              
              --scanIt;
              }

            if ( fromLeft && fromRight )
              {
              scanIt.SetIndex( leftIndex );
              while ( scanIt.GetIndex() != rightIndex )
                {
                if ( scanIt.Get() == 0 )
                  {
		  scanIt.Set( static_cast< unsigned short >( cip::WHOLELUNG ) );
                  continueToAdd = true;
                  }
                ++scanIt;
                }
              }
            }

          //
          // Line Scan the mask image and fill (vertical)
          //
          scanIt.SetDirection( 1 );
          for ( scanIt.GoToBegin(); !scanIt.IsAtEnd(); scanIt.NextLine() )
            {
            fromTop = false;
            fromBottom = false;

            scanIt.GoToBeginOfLine();
            while ( !scanIt.IsAtEndOfLine() )
              {
	      if ( (scanIt.Get() == static_cast< unsigned short >( cip::WHOLELUNG )) && !fromTop )
                {
                fromTop = true;
                topIndex = scanIt.GetIndex();
                }

              ++scanIt;
              }

            //
            // At this point, the iterator can be past the buffer, so
            // scoot it back one spot so our first evaluation of the
            // label map value is actually valid
            //
            --scanIt;
            while ( !scanIt.IsAtReverseEndOfLine() )
              {
	      if ( (scanIt.Get() == static_cast< unsigned short >( cip::WHOLELUNG )) && !fromBottom )
                {
                fromBottom = true;
                bottomIndex = scanIt.GetIndex();
                }

              --scanIt;
              }

            if ( fromTop && fromBottom )
              {
              scanIt.SetIndex(topIndex);
              while ( scanIt.GetIndex() != bottomIndex )
                {
                if ( scanIt.Get() == 0 )
                  {
		  scanIt.Set( static_cast< unsigned short >( cip::WHOLELUNG ) );
                  continueToAdd = true;
                  }
                ++scanIt;
                }
              }
            }
          }
        }
      }

    //
    // Filled images must be rotated back to their original orientations
    //
    SliceIteratorType sIt( slice, slice->GetBufferedRegion() );

    sIt.GoToBegin();
    erIt.GoToBegin();
    while ( !sIt.IsAtEnd() )
      {
      sIt.Set( erIt.Get() );

      ++sIt;
      ++erIt;
      }
      
    double angleInDegrees = -static_cast< float >( numRotations )*degreesResolution;
    angle = angleInDegrees * degreesToRadians;

    TransformType::Pointer transformRec = TransformType::New();
      transformRec->Translate( translation1 );
      transformRec->Rotate2D( angle, false );
      transformRec->Translate( translation2, false );
      
    extractorResampler->SetInput( slice );
    extractorResampler->SetTransform( transformRec );
    extractorResampler->Update();
    
    ImageType::SizeType sliceIteratorSize;
      sliceIteratorSize[0] = imageSize[0];
      sliceIteratorSize[1] = imageSize[1];
      sliceIteratorSize[2] = 1;

    ImageType::IndexType sliceIteratorStartIndex;
      sliceIteratorStartIndex[0] = 0;
      sliceIteratorStartIndex[1] = 0;
      sliceIteratorStartIndex[2] = i;

    ImageType::RegionType sliceIteratorRegion;
      sliceIteratorRegion.SetSize( sliceIteratorSize );
      sliceIteratorRegion.SetIndex( sliceIteratorStartIndex );      

    IteratorType rIt( image, sliceIteratorRegion );

    erIt.GoToBegin();
    rIt.GoToBegin();
    while ( !rIt.IsAtEnd() )
      {
      rIt.Set( erIt.Get() );

      ++erIt;
      ++rIt;
      }
    }
}


void ResampleImage( ImageType::Pointer image, ImageType::Pointer subsampledROIImage, float downsampleFactor )
{
  ImageType::SizeType inputSize = image->GetBufferedRegion().GetSize();

  ImageType::SpacingType inputSpacing = image->GetSpacing();

  ImageType::SpacingType outputSpacing;
    outputSpacing[0] = inputSpacing[0]*downsampleFactor;
    outputSpacing[1] = inputSpacing[1]*downsampleFactor;
    outputSpacing[2] = inputSpacing[2]*downsampleFactor;

  ImageType::SizeType outputSize;
    outputSize[0] = static_cast< unsigned int >( static_cast< double >( inputSize[0] )/downsampleFactor );
    outputSize[1] = static_cast< unsigned int >( static_cast< double >( inputSize[1] )/downsampleFactor );
    outputSize[2] = static_cast< unsigned int >( static_cast< double >( inputSize[2] )/downsampleFactor );

  ImageInterpolatorType::Pointer interpolator = ImageInterpolatorType::New();

  IdentityType::Pointer transform = IdentityType::New();
    transform->SetIdentity();

  ImageResampleType::Pointer resampler = ImageResampleType::New();
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

  subsampledROIImage->SetRegions( resampler->GetOutput()->GetBufferedRegion().GetSize() );
  subsampledROIImage->Allocate();
  subsampledROIImage->FillBuffer( 0 );
  subsampledROIImage->SetSpacing( outputSpacing );
  subsampledROIImage->SetOrigin( image->GetOrigin() );

  IteratorType rIt( resampler->GetOutput(), resampler->GetOutput()->GetBufferedRegion() );
  IteratorType sIt( subsampledROIImage, subsampledROIImage->GetBufferedRegion() );

  rIt.GoToBegin();
  sIt.GoToBegin();
  while ( !sIt.IsAtEnd() )
    {
    sIt.Set( rIt.Get() );
    
    ++rIt;
    ++sIt;
    }
} 


unsigned short GetMaxValueInImage( ImageType::Pointer image )
{
  unsigned short maxValue = 0;

  IteratorType it( image, image->GetBufferedRegion() );

  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( it.Get() > maxValue )
      {
      maxValue = it.Get();
      }

    ++it;
    }

  return maxValue;
}
    

#endif
