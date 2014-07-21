/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads atlas lung images and generates a
 *  convex hull image corresponding to them. It is assumed that the
 *  atlas exists as two separate atlases: one for the left lung and
 *  one for the right. It is also assumed that the the maximum value
 *  in each corresponds to a probability of 1 and the value 0
 *  corresponds to a probability of 0.
 *
 *  The algorithm proceeds by reading in the left atlas and
 *  thresholding according to a specified probability threhold (a
 *  float-valued quantity ranging from 0 to 1). The right atlas is
 *  read in and similarly thresholded. The union of the two images is
 *  created, and the resulting image is downsampled for faster
 *  processing. After downsampling, the convex hull is created. The
 *  convex hull is represented as a binary image (0 = background, 1 =
 *  foreground). The convex hull is upsampled so that it has the same
 *  extent as the original image, and it is then written to file.
 *
 *  USAGE:
 *
 *  GenerateAtlasConvexHull.exe [-p \<float\>] [-s \<float\>]
 *                              [-d \<float\>] [-n \<int\>] 
 *                              [-o \<string\>] -r \<string\> -l \<string\>
 *                              [--] [--version] [-h]
 *
 * Where:
 *
 *   -p \<float\>,  --probability \<float\>
 *     Probability threshold in the interval [0,1] (default is 0.5).This
 *     parameter controls the level at which the atlas is thresholded prior
 *     to convex hull creation
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
 *   -r \<string\>,  --rightAtlas \<string\>
 *     (required)  Right lung atlas file name
 *
 *   -l \<string\>,  --leftAtlas \<string\>
 *     (required)  Left lung atlas file name
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
 *  $Date: 2012-09-05 17:02:14 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 232 $
 *  $Author: jross $
 *
 */




#include "cipChestConventions.h"
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
#include "GenerateAtlasConvexHullCLP.h"

namespace
{
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

/*unsigned short GetMaxValueInImage( ImageType::Pointer );
void ReassignImageToConvexHull( ImageType::Pointer, int, float );
void ResampleImage( ImageType::Pointer, ImageType::Pointer, float );
*/

} //end namespace

int main( int argc, char *argv[] )
{

	  PARSE_ARGS;
  //
  // Begin by defining the arguments to be passed
  //
  /*std::string leftAtlasFileName;
  std::string rightAtlasFileName;
  std::string outputFileName;
  int         numRotations         = 1;
  float       degreesResolution    = 45.0;
  float       downsampleFactor     = 1.0;
  float       probabilityThreshold = 0.5;*/

   ImageType::Pointer completeThresholdedAtlas = ImageType::New();

  {
  //
  // Read the left atlas. 
  //
  std::cout << "Reading left atlas..." << std::endl;
  ReaderType::Pointer leftReader = ReaderType::New();
    leftReader->SetFileName( leftAtlasFileName.c_str() ); //.c_str()
  try
    {
    leftReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading left atlas:";
    std::cerr << excp << std::endl;

    return cip::ATLASREADFAILURE;
    }

  unsigned short maxValue = GetMaxValueInImage( leftReader->GetOutput() );

  completeThresholdedAtlas->SetRegions( leftReader->GetOutput()->GetBufferedRegion().GetSize() );
  completeThresholdedAtlas->Allocate();
  completeThresholdedAtlas->FillBuffer( 0 );
  completeThresholdedAtlas->SetSpacing( leftReader->GetOutput()->GetSpacing() );
  completeThresholdedAtlas->SetOrigin( leftReader->GetOutput()->GetOrigin() );

  IteratorType lIt( leftReader->GetOutput(), leftReader->GetOutput()->GetBufferedRegion() );
  IteratorType it( completeThresholdedAtlas, completeThresholdedAtlas->GetBufferedRegion() );

  lIt.GoToBegin();
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( lIt.Get() >= static_cast< double >( maxValue )*probabilityThreshold )
      {
      it.Set( 1 );
      }

    ++lIt;
    ++it;
    }
  }

  {
  //
  // Read the right atlas
  //
  std::cout << "Reading right atlas..." << std::endl;
  ReaderType::Pointer rightReader = ReaderType::New();
    rightReader->SetFileName( rightAtlasFileName.c_str() ); //.c_str()
  try
    {
    rightReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading right atlas:";
    std::cerr << excp << std::endl;

    return cip::ATLASREADFAILURE;
    }

  unsigned short maxValue = GetMaxValueInImage( rightReader->GetOutput() );

  IteratorType rIt( rightReader->GetOutput(), rightReader->GetOutput()->GetBufferedRegion() );
  IteratorType it( completeThresholdedAtlas, completeThresholdedAtlas->GetBufferedRegion() );

  rIt.GoToBegin();
  it.GoToBegin();
  while ( !it.IsAtEnd() )
    {
    if ( rIt.Get() >= static_cast< double >( maxValue )*probabilityThreshold )
      {
      it.Set( 1 );
      }

    ++rIt;
    ++it;
    }
  }

  //
  // Before computing the convex hull, subsample the image
  //
  ImageType::Pointer subSampledThresholdedAtlas = ImageType::New();

  std::cout << "Subsampling atlas..." << std::endl;
  ResampleImage( completeThresholdedAtlas, subSampledThresholdedAtlas, downsampleFactor );

  //
  // Now compute the convex hull
  //
  std::cout << "Computing convex hull..." << std::endl;
  ReassignImageToConvexHull( subSampledThresholdedAtlas, numRotations, degreesResolution );  

  //
  // Up-sample the image
  //
  ImageType::Pointer convexHullImage = ImageType::New();

  std::cout << "Up-sampling atlas..." << std::endl;
  ResampleImage( subSampledThresholdedAtlas, convexHullImage, 1.0/downsampleFactor );

  //
  // Write the convex hull
  //
  std::cout << "Writing convex hull... " << std::endl;
  WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputFileName.c_str() ); //
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

