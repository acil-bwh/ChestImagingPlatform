/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to generate fissure shape models based on a
 *  training set. The basic approach is as follows: a reference image
 *  is read and then registered to the label map for which the user
 *  wants to produce fissure models. Additionally, a set of training
 *  data is read in. These data sets consist of fissure indices and a
 *  transform file that indicates the mapping between the training data
 *  label map and the reference label map. Once the reference image is
 *  registered to the input image, the resulting transform in
 *  conjunction with the training data transform is used to map the
 *  training fissure points to the input image's coordinate frame. This
 *  is done repeatedly for all the training data sets. Once all the
 *  fissure points are in the coordinate frame of the input image, PCA
 *  is performed to get the mean and modes of the fissure points. This
 *  data is then printed to file.
 * 
 *  The reference image for model building has been (arbitrarily)
 *  chosen to be the COPDGene case, 10002K_INSP_STD_BWH_COPD. There are 
 *  20 training datasets consisting, including INSP and EXP data. These 
 *  datasets were manually segmented by George Washko and Alejandro Diaz 
 *  for the MICCAI '09 submission. The transform files used to map the 
 *  training data to the reference dataset and the files for the training 
 *  data can be found in ~/Processed/Atlases/LungLobeAtlases.
 *  
 *  The output file format is as follows: the first line contains the
 *  origin of the input image. The second line contains the spacing of
 *  the input image. The third line indicates the number of modes,
 *  N. The fourth line is the mean vector. The entries are for the
 *  continuous index z-values of the control points. The next N lines
 *  contain the mode variances and a weight value. Upon generation of
 *  the shape model file, the weight value will be 0. Later in the lobe
 *  segmentation process, this value can be overwritten with a weight
 *  has been found by fitting to particles data. The next N lines are
 *  the modes themselves -- again, the modes represent the variation of
 *  the control points in the z-direction (the range of our TPS
 *  function). The length of each mode (and mean) vector is m, where m
 *  indicates the number of control points. Finally, the next m lines
 *  indicate the control point indices (the domain locations). Note
 *  that the z-value for these entries is 0. This is to emphasize that
 *  they are in the domain -- the x-y plane.
 * 
 *  USAGE: 
 *
 *  GenerateFissureShapeModels  [-n \<unsigned int\>] [-d \<float\>] 
 *                              [-t \<float\>] [--min \<float\>] [--max \<float\>]
 *                              [--ot \<string\>] [--olo \<string\>] 
 *                              [--oro \<string\>] [--orh \<string\>] --refps
 *                              \<string\> --refim \<string\> -i \<string\>
 *                               --trainPoints \<string\> ...  --trainTrans
 *                              \<string\> ...  [--] [--version] [-h]
 *
 *  Where: 
 *
 *  -n \<unsigned int\>,  --numIters \<unsigned int\>
 *    Number of iterations. Default is 20
 *
 *  -d \<float\>,  --down \<float\>
 *    Down sample factor. The fixed and moving images will be down sampled
 *    by this amount before registration. (Default is 1.0, i.e. no down
 *    sampling).
 *
 *  -t \<float\>,  --transScale \<float\>
 *    Translation scale. Default is 0.001
 *
 *  --min \<float\>
 *    Min step length. Default is 0.001
 *
 *  --max \<float\>
 *    Max step length. Default is 1.0
 *
 *  --ot \<string\>
 *    Output reference image to input label map transform file name
 *
 *  --olo \<string\>
 *    Output left oblique shape model file name
 *
 *  --oro \<string\>
 *    Output right oblique shape model file name
 *
 *  --orh \<string\>
 *    Output right horizontal shape model file name
 *
 *  --refps \<string\>
 *    (required)  Reference points corresponding to the reference image.
 *
 *  --refim \<string\>
 *    (required)  Reference image label map corresponding to the reference
 *    points. This is the image to which all other image data in the
 *    training set is registered to
 *
 *  -i \<string\>,  --inFileName \<string\>
 *    (required)  Input label map to serve as starting point for lung lobe
 *    segmentation .The level of lung segmentation 'grunularity' should be
 *    at the left lung right lung split level. In other words, a mask
 *    divided into thirds (left and right) will work fine, as will one for
 *    which only the left and right lung are labeled.
 *
 *  --trainPoints \<string\>  (accepted multiple times)
 *    (required)  Region and type points file corresponding to the
 *    previously called transform file (see notes for --trainTrans)
 *
 *  --trainTrans \<string\>  (accepted multiple times)
 *    (required)  Training data set transform file. This file contains the
 *    transform that is used to map the corresponding training data set's
 *    fissure points to the reference image's coordinate system. It should
 *    immediately be followed with a region and type points file (specified
 *    with the --trainPoints flag)
 *
 *  --,  --ignore_rest
 *    Ignores the rest of the labeled arguments following this flag.
 *
 *  --version
 *    Displays version information and exits.
 *
 *  -h,  --help
 *    Displays usage information and exits.
 *
 *  $Date: 2012-09-08 21:08:55 -0400 (Sat, 08 Sep 2012) $
 *  $Revision: 251 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipConventions.h"
#include "cipThinPlateSplineSurface.h"
#include "itkNrrdImageIO.h"
#include "itkImage.h"
#include "itkTransformFileReader.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegistrationMethod.h"
#include "itkCenteredTransformInitializer.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkIdentityTransform.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkContinuousIndex.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "cipLobeSurfaceModel.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipHelper.h"
#include "GenerateLobeSurfaceModelsCLP.h"

typedef itk::Image< unsigned short, 3 >                                             ImageType;
typedef itk::Image< unsigned short, 2 >                                             ImageSliceType;
typedef itk::ResampleImageFilter< ImageType, ImageType >                            ResampleFilterType;
typedef itk::ImageFileReader< ImageType >                                           ImageReaderType;
typedef itk::ImageFileWriter< ImageType >                                           ImageWriterType;
typedef itk::ImageFileWriter< ImageSliceType >                                      SliceWriterType;
typedef itk::RegularStepGradientDescentOptimizer                                    OptimizerType;
typedef itk::ImageRegistrationMethod< ImageType, ImageType >                        RegistrationType;
typedef itk::KappaStatisticImageToImageMetric< ImageType, ImageType >               MetricType;
typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double >           InterpolatorType;
typedef itk::AffineTransform< double, 3 >                                           TransformType;
typedef itk::CenteredTransformInitializer< TransformType, ImageType, ImageType >    InitializerType;
typedef OptimizerType::ScalesType                                                   OptimizerScalesType;
typedef itk::ImageRegionIteratorWithIndex< ImageType >                              IteratorType;
typedef itk::ImageRegionIteratorWithIndex< ImageSliceType >                         SliceIteratorType;
typedef itk::ResampleImageFilter< ImageType, ImageType >                            ResampleType;
typedef itk::IdentityTransform< double, 3 >                                         IdentityType;
typedef itk::CIPExtractChestLabelMapImageFilter                                     LabelMapExtractorType;
typedef itk::TransformFileReader::TransformListType*                                TransformListType;
typedef itk::ContinuousIndex< double, 3 >                                           ContinuousIndexType;

struct PCA
{
  // Eigenvalues are in descending order
  std::vector< double >                    meanVec;
  std::vector< double >                    modeVec;
  std::vector< std::vector< double > >     modeVecVec;
  unsigned int                             numModes;
};


void ResampleImage( ImageType::Pointer, ImageType::Pointer, float );
TransformType::Pointer RegisterLungs( ImageType::Pointer, ImageType::Pointer, float, float, float, int );
void WriteTransformToFile( TransformType::Pointer, std::string );
TransformType::Pointer GetTransformFromFile( const char* );
void ReadFissurePointsFromFile( std::string, std::vector< ImageType::PointType >*, std::vector< ImageType::PointType >*, std::vector< ImageType::PointType >* );
void GetDomainPoints( std::vector< std::vector< ImageType::PointType > >, std::vector< ImageType::PointType >*, ImageType::Pointer, int );
void GetZValuesFromTPS( std::vector< ImageType::PointType >, std::vector< double >*, std::vector< ImageType::PointType >,
                        ImageType::Pointer );
PCA GetShapePCA( std::vector< std::vector< double > > );
void GetDomainPatternedPoints(ImageType::PointType, float[2], unsigned int, std::vector< ImageType::PointType >* );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  if ( trainTransformFileVec.size() != trainPointsFileVec.size() )
    {
      std::cerr << "Must specify same number of training data transform files ";
      std::cerr << "and training data points files" << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }
  if ( trainTransformFileVec.size() == 0 )
    {
      std::cerr << "Must supply training data" << std::endl;
      return cip::ARGUMENTPARSINGERROR;
    }

  // The reference image for model building has been (arbitrarily)
  // chosen to be the COPDGene case, 10002K_INSP_STD_BWH_COPD. There are
  // 20 training datasets consisting, including INSP and EXP data. These
  // datasets were manually segmented by George Washko and Alejandro Diaz
  // for the MICCAI 09 submission. The transform files used to map the
  // training data to the reference dataset and the files for the training
  // data can be found in ~/Processed/Atlases/LungLobeAtlases

  unsigned int numTrainingSets = trainTransformFileVec.size();

  std::cout << "Reading fixed image..." << std::endl;
  ImageReaderType::Pointer fixedReader = ImageReaderType::New();
    fixedReader->SetFileName( inputFileName );
  try
    {
    fixedReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while updating fixed reader:";
    std::cerr << excp << std::endl;
    }

  // Now that we've read in the left-lung right-lung label map, we can create a set of
  // points in the axial plane (the "domain" points) at which we'll compute TPS values.
  // The domain points will be spread across the bounding box region in the axial plane;
  // there will be a set of domain points for the right lung and a separate set of 
  // domain points for the right lung. The right oblique and right horizontal lobe 
  // boundaries will share the same set of domain points for simplicity. Start by 
  // obtaining the domain points for the left lung.
  cip::LabelMapType::RegionType leftLungBoundingBox = 
    cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(fixedReader->GetOutput(), (unsigned char)(cip::LEFTLUNG));

  ImageType::PointType leftBoundingBoxStartPoint;
  fixedReader->GetOutput()->TransformIndexToPhysicalPoint( leftLungBoundingBox.GetIndex(), leftBoundingBoxStartPoint );

  float leftWidth[2];
    leftWidth[0] = leftLungBoundingBox.GetSize()[0]*fixedReader->GetOutput()->GetSpacing()[0];
    leftWidth[1] = leftLungBoundingBox.GetSize()[1]*fixedReader->GetOutput()->GetSpacing()[1];

  std::vector< ImageType::PointType > leftDomainPatternPoints;
  GetDomainPatternedPoints(leftBoundingBoxStartPoint, leftWidth, 2, &leftDomainPatternPoints );

  cip::LabelMapType::RegionType rightLungBoundingBox = 
    cip::GetLabelMapChestRegionChestTypeBoundingBoxRegion(fixedReader->GetOutput(), (unsigned char)(cip::RIGHTLUNG));

  // Now ge the set of domain points for the right lung
  ImageType::PointType rightBoundingBoxStartPoint;
  fixedReader->GetOutput()->TransformIndexToPhysicalPoint( rightLungBoundingBox.GetIndex(), rightBoundingBoxStartPoint );

  float rightWidth[2];
    rightWidth[0] = rightLungBoundingBox.GetSize()[0]*fixedReader->GetOutput()->GetSpacing()[0];
    rightWidth[1] = rightLungBoundingBox.GetSize()[1]*fixedReader->GetOutput()->GetSpacing()[1];

  std::vector< ImageType::PointType > rightDomainPatternPoints;
  GetDomainPatternedPoints(rightBoundingBoxStartPoint, rightWidth, 2, &rightDomainPatternPoints );

  // We need to get the transform that maps the reference image (from
  // our training set) to the input image (whose lobes we want to
  // segment). We read both these images, downsample them, and then
  // register them. Once this has been done, we can use the resulting
  // transform to map all training points to the input image's
  // coordinate frame
  TransformType::Pointer refToInputTransform = TransformType::New();

  ImageType::Pointer subSampledFixedImage = ImageType::New();

  std::cout << "Subsampling fixed image..." << std::endl;
  ResampleImage( fixedReader->GetOutput(), subSampledFixedImage, downsampleFactor );

  ImageType::Pointer subSampledMovingImage = ImageType::New();

  std::cout << "Reading moving image..." << std::endl;
  ImageReaderType::Pointer movingReader = ImageReaderType::New();
    movingReader->SetFileName( refImageFileName );
  try
    {
      movingReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught while updating moving reader:";
      std::cerr << excp << std::endl;
    }
    
  std::cout << "Subsampling moving image..." << std::endl;
  ResampleImage( movingReader->GetOutput(), subSampledMovingImage, downsampleFactor );

  // Now extract the whole lung region from the resized fixed and
  // moving images
  std::cout << "Extracting whole lung region from down sampled fixed image..." << std::endl;
  LabelMapExtractorType::Pointer fixedExtractor = LabelMapExtractorType::New();
    fixedExtractor->SetInput( subSampledFixedImage );
    fixedExtractor->SetChestRegion( static_cast< unsigned char >( cip::WHOLELUNG ) );
    fixedExtractor->Update();

  std::cout << "Extracting whole lung region from down sampled moving image..." << std::endl;
  LabelMapExtractorType::Pointer movingExtractor = LabelMapExtractorType::New();
    movingExtractor->SetInput( subSampledMovingImage );
    movingExtractor->SetChestRegion( static_cast< unsigned char >( cip::WHOLELUNG ) );
    movingExtractor->Update();

  std::cout << "Registering reference image to input image..." << std::endl;
  refToInputTransform = RegisterLungs( fixedExtractor->GetOutput(), movingExtractor->GetOutput(), 
  				       maxStepLength, minStepLength, translationScale, numberOfIterations );

  if ( outRefToInputTransformFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Writing reference to label map transform..." << std::endl;
      WriteTransformToFile( refToInputTransform, outRefToInputTransformFileName );
    }

  // Now that we have the transform that maps the reference image to
  // the input image, we need to read in and map all the training
  // points so that they are in the input image's coordinate frame.
  // The following three vectors of vectors will contain all the
  // points from the training set (transformed to be in the input
  // image's coordinate frame)
  std::vector< std::vector< ImageType::PointType > > loTransPointsVecVec;
  std::vector< std::vector< ImageType::PointType > > roTransPointsVecVec;
  std::vector< std::vector< ImageType::PointType > > rhTransPointsVecVec;

  for ( unsigned int i=0; i<numTrainingSets; i++ )
    {
    // First read the transform
    std::cout << "Reading transform from file..." << std::endl;
    TransformType::Pointer transform = GetTransformFromFile( trainTransformFileVec[i].c_str() );

    // Now read in the points
    std::vector< ImageType::PointType > loPointsVec;
    std::vector< ImageType::PointType > roPointsVec;
    std::vector< ImageType::PointType > rhPointsVec;
    std::cout << "Reading fissure points from file..." << std::endl;
    ReadFissurePointsFromFile( trainPointsFileVec[i].c_str(), &loPointsVec, &roPointsVec, &rhPointsVec );    

    // Now that the points have been read in, we must transform them
    // so that they are in the coordinate frame of the input
    // image. To do this, we must first transform them so that they
    // are in the reference image's coordinate frame, and then
    // transform them so that they wind up in the input image's
    // coordinate frame
    for ( unsigned lo=0; lo<loPointsVec.size(); lo++ )
      {
      loPointsVec[lo] = transform->TransformPoint( loPointsVec[lo] );
      loPointsVec[lo] = refToInputTransform->TransformPoint( loPointsVec[lo] );
      }
    loTransPointsVecVec.push_back( loPointsVec );

    for ( unsigned ro=0; ro<roPointsVec.size(); ro++ )
      {
      roPointsVec[ro] = transform->TransformPoint( roPointsVec[ro] );
      roPointsVec[ro] = refToInputTransform->TransformPoint( roPointsVec[ro] );
      }
    roTransPointsVecVec.push_back( roPointsVec );

    for ( unsigned rh=0; rh<rhPointsVec.size(); rh++ )
      {
      rhPointsVec[rh] = transform->TransformPoint( rhPointsVec[rh] );
      rhPointsVec[rh] = refToInputTransform->TransformPoint( rhPointsVec[rh] );
      }
    rhTransPointsVecVec.push_back( rhPointsVec );
    }

  // Now we have the domain locations for each of our three
  // boundaries. To build our PCA model, we need to evaluate the z
  // values at the domain locations for each of our training 
  // datasets. THe right oblique ('ro') and right horizontal ('rh')
  // are evaluated separately, but then collected ('right').
  std::cout << "Getting range locations..." << std::endl;
  std::vector< std::vector< double > > loRangeValuesVecVec;
  std::vector< std::vector< double > > rightRangeValuesVecVec;

  for ( unsigned int i=0; i<numTrainingSets; i++ )
    {
    std::vector< double > roRangeValuesVec;
    std::vector< double > rhRangeValuesVec;
    std::vector< double > loRangeValuesVec;
    std::vector< double > rightRangeValuesVec;

    GetZValuesFromTPS( rightDomainPatternPoints, &roRangeValuesVec, roTransPointsVecVec[i], fixedReader->GetOutput() );
    GetZValuesFromTPS( rightDomainPatternPoints, &rhRangeValuesVec, rhTransPointsVecVec[i], fixedReader->GetOutput() );
    GetZValuesFromTPS( leftDomainPatternPoints, &loRangeValuesVec, loTransPointsVecVec[i], fixedReader->GetOutput() );

    for ( unsigned int j=0; j<roRangeValuesVec.size(); j++ )
      {
	rightRangeValuesVec.push_back( roRangeValuesVec[j] );
      }
    for ( unsigned int j=0; j<rhRangeValuesVec.size(); j++ )
      {
	rightRangeValuesVec.push_back( rhRangeValuesVec[j] );
      }

    loRangeValuesVecVec.push_back( loRangeValuesVec );
    rightRangeValuesVecVec.push_back( rightRangeValuesVec );
    }

  // Now we have all the training information needed for building the
  // PCA-based fissure models for this case. 
  PCA loShapePCA    = GetShapePCA( loRangeValuesVecVec );
  PCA rightShapePCA = GetShapePCA( rightRangeValuesVecVec );

  // Now create shape models for each of the three lobe boundaries. To do this we need to 
  // collect the domain locations and mean vector into a single collection of points
  // for each of the three boundaries
  std::vector< double* >* rightMeanSurfacePoints = new std::vector< double* >;
  for ( unsigned int i=0; i<rightShapePCA.meanVec.size(); i++ )
    {
      unsigned int index = i%(rightShapePCA.meanVec.size()/2);

      double* point = new double[3];
        point[0] = rightDomainPatternPoints[index][0];
  	point[1] = rightDomainPatternPoints[index][1];
  	point[2] = rightShapePCA.meanVec[i];

      rightMeanSurfacePoints->push_back( point );
    }

  std::vector< double* >* loMeanSurfacePoints = new std::vector< double* >;
  for ( unsigned int i=0; i<loShapePCA.meanVec.size(); i++ )
    {
      double* point = new double[3];
        point[0] = leftDomainPatternPoints[i][0];
  	point[1] = leftDomainPatternPoints[i][1];
  	point[2] = loShapePCA.meanVec[i];

      loMeanSurfacePoints->push_back( point );
    }

  // Now create the boundary models
  std::vector< double > rightModeWeights;
  for ( unsigned int n=0; n<rightShapePCA.modeVec.size(); n++ )
    {
      rightModeWeights.push_back( 0.0 );
    }

  std::vector< double > loModeWeights;
  for ( unsigned int n=0; n<loShapePCA.modeVec.size(); n++ )
    {
      loModeWeights.push_back( 0.0 );
    }

  double* origin = new double[3];
    origin[0] = fixedReader->GetOutput()->GetOrigin()[0];
    origin[1] = fixedReader->GetOutput()->GetOrigin()[1];
    origin[2] = fixedReader->GetOutput()->GetOrigin()[2];

  double* spacing = new double[3];
    spacing[0] = fixedReader->GetOutput()->GetSpacing()[0];
    spacing[1] = fixedReader->GetOutput()->GetSpacing()[1];
    spacing[2] = fixedReader->GetOutput()->GetSpacing()[2];

  cipLobeSurfaceModel* rightShapeModel = new cipLobeSurfaceModel();
    rightShapeModel->SetImageOrigin( origin );
    rightShapeModel->SetImageSpacing( spacing );
    rightShapeModel->SetMeanSurfacePoints( rightMeanSurfacePoints );
    rightShapeModel->SetEigenvalues( &rightShapePCA.modeVec );
    rightShapeModel->SetModeWeights( &rightModeWeights );
    rightShapeModel->SetEigenvectors( &rightShapePCA.modeVecVec );
    rightShapeModel->SetNumberOfModes( rightShapePCA.numModes );

  cipLobeSurfaceModel* loShapeModel = new cipLobeSurfaceModel();
    loShapeModel->SetImageOrigin( origin );
    loShapeModel->SetImageSpacing( spacing );
    loShapeModel->SetMeanSurfacePoints( loMeanSurfacePoints );
    loShapeModel->SetModeWeights( &loModeWeights );
    loShapeModel->SetEigenvalues( &loShapePCA.modeVec );
    loShapeModel->SetEigenvectors( &loShapePCA.modeVecVec );
    loShapeModel->SetNumberOfModes( loShapePCA.numModes );

  if ( rightShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right shape model to file..." << std::endl;
    cip::LobeSurfaceModelIO rightWriter;
      rightWriter.SetFileName( rightShapeModelFileName );
      rightWriter.SetInput( rightShapeModel );
      rightWriter.Write();
    }

  if ( leftShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing left shape model to file..." << std::endl;
    cip::LobeSurfaceModelIO loWriter;
      loWriter.SetFileName( leftShapeModelFileName );
      loWriter.SetInput( loShapeModel );
      loWriter.Write();
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


void WriteTransformToFile( TransformType::Pointer transform, std::string fileName )
{
  itk::TransformFileWriter::Pointer transformWriter = itk::TransformFileWriter::New();
    transformWriter->SetInput( transform );
    transformWriter->SetFileName( fileName );
  try
    {
    transformWriter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while updating transform writer:";
    std::cerr << excp << std::endl;
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

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  IdentityType::Pointer transform = IdentityType::New();
    transform->SetIdentity();

  ResampleType::Pointer resampler = ResampleType::New();
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


TransformType::Pointer RegisterLungs( ImageType::Pointer fixedImage, ImageType::Pointer movingImage, 
                                      float maxStepLength, float minStepLength, float translationScale, int numberOfIterations )
{
  cip::ChestConventions conventions;
  
  unsigned short foregroundValue = conventions.GetValueFromChestRegionAndType( static_cast< unsigned char >( cip::WHOLELUNG ), 
									       static_cast< unsigned char >( cip::UNDEFINEDTYPE) );

  MetricType::Pointer metric = MetricType::New();
    metric->ComplementOn();
    metric->SetForegroundValue( foregroundValue );

  TransformType::Pointer transform = TransformType::New();

  InitializerType::Pointer initializer = InitializerType::New();
    initializer->SetTransform( transform );
    initializer->SetFixedImage(  fixedImage );
    initializer->SetMovingImage( movingImage );
    initializer->MomentsOn();
    initializer->InitializeTransform();

  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
    optimizerScales[0] =  1.0;   optimizerScales[1] =  1.0;   optimizerScales[2] =  1.0;
    optimizerScales[3] =  1.0;   optimizerScales[4] =  1.0;   optimizerScales[5] =  1.0;
    optimizerScales[6] =  1.0;   optimizerScales[7] =  1.0;   optimizerScales[8] =  1.0;
    optimizerScales[9]  =  translationScale;
    optimizerScales[10] =  translationScale;
    optimizerScales[11] =  translationScale;

  OptimizerType::Pointer optimizer = OptimizerType::New();
    optimizer->SetScales( optimizerScales );
    optimizer->SetMaximumStepLength( maxStepLength );
    optimizer->SetMinimumStepLength( minStepLength ); 
    optimizer->SetNumberOfIterations( numberOfIterations );

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  RegistrationType::Pointer registration = RegistrationType::New();  
    registration->SetMetric( metric );
    registration->SetOptimizer( optimizer );
    registration->SetInterpolator( interpolator );
    registration->SetTransform( transform );
    registration->SetFixedImage( fixedImage );
    registration->SetMovingImage( movingImage );
    registration->SetFixedImageRegion( fixedImage->GetBufferedRegion() );
    registration->SetInitialTransformParameters( transform->GetParameters() );
  try 
    { 
    registration->Update(); 
    } 
  catch( itk::ExceptionObject &excp ) 
    { 
    std::cerr << "ExceptionObject caught while executing registration" << std::endl; 
    std::cerr << excp << std::endl; 
    } 

  OptimizerType::ParametersType finalParams = registration->GetLastTransformParameters();

  TransformType::Pointer finalTransform = TransformType::New();
    finalTransform->SetParameters( finalParams );
    finalTransform->SetCenter( transform->GetCenter() );
    finalTransform->GetInverse( finalTransform );

  return finalTransform;
}


TransformType::Pointer GetTransformFromFile( const char* fileName )
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
  
  TransformListType transformList = transformReader->GetTransformList();

  itk::TransformFileReader::TransformListType::const_iterator it;
  
  it = transformList->begin();

  TransformType::Pointer transform = static_cast< TransformType* >( (*it).GetPointer() ); 

  transform->GetInverse( transform );

  return transform;
}


void ReadFissurePointsFromFile( const std::string fileName, 
                                std::vector< ImageType::PointType >* leftObliquePointsVec, 
                                std::vector< ImageType::PointType >* rightObliquePointsVec, 
                                std::vector< ImageType::PointType >* rightHorizontalPointsVec )
{
  cipChestRegionChestTypeLocationsIO regionsTypesIO;
    regionsTypesIO.SetFileName( fileName );
    regionsTypesIO.Read();

  for ( unsigned int i=0; i<regionsTypesIO.GetOutput()->GetNumberOfTuples(); i++ )
    {      
      unsigned char cipRegion = regionsTypesIO.GetOutput()->GetChestRegionValue( i );
      unsigned char cipType   = regionsTypesIO.GetOutput()->GetChestTypeValue( i );

      double* point = new double[3];
      regionsTypesIO.GetOutput()->GetLocation( i, point );
      ImageType::PointType itkPoint;
        itkPoint[0] = point[0];
	itkPoint[1] = point[1];
	itkPoint[2] = point[2];

      if ( cipRegion == static_cast< unsigned char >( cip::LEFTLUNG ) ||
	   cipRegion == static_cast< unsigned char >( cip::LEFTSUPERIORLOBE ) ||
	   cipRegion == static_cast< unsigned char >( cip::LEFTINFERIORLOBE ) )
	{
	  if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
	    {
	      (*leftObliquePointsVec).push_back( point ); 
	    }
	}
      else if ( cipRegion == static_cast< unsigned char >( cip::RIGHTLUNG ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) )
	{
	  if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
	    {
	      (*rightObliquePointsVec).push_back( point ); 
	    }
	  else if ( cipType == static_cast< unsigned char >( cip::HORIZONTALFISSURE ) )
	    {
	      (*rightHorizontalPointsVec).push_back( point ); 
	    }
	}
    }  
}


//
// This function considers a bounding box in the axial plane an produces a collection of
// points distributed evenly throughout the bounding box region and along the bounding
// box border. The 'density' parameter controls how many points are used to sample the
// region. At a minimum, there will be a point at each corner of the bounding box and
// a point at the center. If 'density' is set to 1, there will additionally be another
// point centered in the middle of each edge segment and one point centered in the 
// middle of each segment connecting the center and each point along the border. If
// 'density' is set to 2, there will be two points evenly spaced along each border
// segment and two points evenly spaced along each segment connecting the center the
// center point to each border point, etc.
//
void GetDomainPatternedPoints(ImageType::PointType start, float width[2], unsigned int density,
			      std::vector< ImageType::PointType >* patternPoints)
{
  // Create the center point and add it
  ImageType::PointType center;
    center[0] = start[0] + width[0]/2.0;
    center[1] = start[1] + width[1]/2.0;
    center[2] = 0.0;
  
  (*patternPoints).push_back( center );

  // Now create the corner points and add them. Note that the corners are
  // ordered in clockwise fashion. This is important for the code below.
  std::vector< ImageType::PointType > corners;
  std::vector< ImageType::PointType > borderPoints;

  ImageType::PointType corner1;
    corner1[0] = start[0];
    corner1[1] = start[1];
    corner1[2] = 0.0;
  (*patternPoints).push_back( corner1 );
  corners.push_back( corner1 );

  ImageType::PointType corner2;
    corner2[0] = start[0] + width[0];
    corner2[1] = start[1];
    corner2[2] = 0.0;
  (*patternPoints).push_back( corner2 );
  corners.push_back( corner2 );

  ImageType::PointType corner3;
    corner3[0] = corner2[0];
    corner3[1] = corner2[1] + width[1];
    corner3[2] = 0.0;
  (*patternPoints).push_back( corner3 );
  corners.push_back( corner3 );

  ImageType::PointType corner4;
    corner4[0] = start[0];
    corner4[1] = start[1] + width[1];
    corner4[2] = 0.0;
  (*patternPoints).push_back( corner4 );
  corners.push_back( corner4 );

  // If there is a non-zero density specified, we'll add the edge points
  // and "spoke" points. Start with the edge points
  for ( unsigned int i=0; i<4; i++ )
    {
      borderPoints.push_back( corners[i] );

      double vec[2];
      if ( i < 3 )
	{
        vec[0] = corners[i+1][0] - corners[i][0];
	vec[1] = corners[i+1][1] - corners[i][1];      
	}
      else
	{
        vec[0] = corners[0][0] - corners[3][0];
	vec[1] = corners[0][1] - corners[3][1];      
	}

      for ( unsigned int j=1; j<=density; j++ )
	{
	  ImageType::PointType point;

	  point[0] = corners[i][0] + double(j)*vec[0]/double(density + 1);
	  point[1] = corners[i][1] + double(j)*vec[1]/double(density + 1);
	  point[2] = 0.0;

	  borderPoints.push_back( point );
	  (*patternPoints).push_back( point );	  
	}
    }

  // Now add the "spoke" points if necessary
  for ( unsigned int i=0; i<borderPoints.size(); i++ )
    {
      double vec[2];
        vec[0] = borderPoints[i][0] - center[0];
  	vec[1] = borderPoints[i][1] - center[1];
      
      for ( unsigned int j=1; j<=density; j++ )
  	{
  	  ImageType::PointType point;

  	  point[0] = center[0] + double(j)*vec[0]/double(density + 1);
  	  point[1] = center[1] + double(j)*vec[1]/double(density + 1);
  	  point[2] = 0.0;

  	  (*patternPoints).push_back( point );	  
  	}
    }
}


//
// This function will gather a set of domain indices for a given
// fissure. It does this by projecting all the training points (which
// should be in the input image's coordinate frame at this point) onto
// the x-y plane (this converts the points to indices). After the
// projection, many of the projected points are erased -- this to
// ensure that we don't have too many (unnecessary) indices in our
// domain. The "erase neighborhood" around a given point continues to
// increase until no more than 100 points remain. Allowing too many
// points makes the TPS model too complex, and this severly slows down
// the fitting process.
//
void GetDomainPoints( std::vector< std::vector< ImageType::PointType > > transPointsVecVec, 
                      std::vector< ImageType::PointType >* domainVec, ImageType::Pointer image, int fooWhichFissure )
{

  ImageType::SpacingType spacing = image->GetSpacing();
  ImageType::PointType   origin  = image->GetOrigin();

  ImageType::SizeType size = image->GetBufferedRegion().GetSize();

  ImageSliceType::SizeType sliceSize;
    sliceSize[0] = size[0];
    sliceSize[1] = size[1];

  ImageSliceType::Pointer domainSlice = ImageSliceType::New();
    domainSlice->SetRegions( sliceSize );
    domainSlice->Allocate();
    domainSlice->FillBuffer( 0 );

  ImageType::IndexType       index;
  ImageSliceType::IndexType  sliceIndex;
  ImageType::PointType       point;

  for ( unsigned int d=0; d<transPointsVecVec.size(); d++ )
    {
    for ( unsigned int p=0; p<(transPointsVecVec[d]).size(); p++ )
      {
      image->TransformPhysicalPointToIndex( (transPointsVecVec[d])[p], index );

      sliceIndex[0] = index[0];
      sliceIndex[1] = index[1];
      if ( domainSlice->GetBufferedRegion().IsInside( sliceIndex ) )
        {
        domainSlice->SetPixel( sliceIndex, 1 );
        }
      }
    }

  ImageSliceType::SizeType   roiSize;
  ImageSliceType::RegionType roiRegion;
  ImageSliceType::IndexType  roiStart;

  SliceIteratorType it( domainSlice, domainSlice->GetBufferedRegion() );

  int roiRadius = 6;

  do
    {
    (*domainVec).clear();

    roiRadius += 1;

    it.GoToBegin();
    while ( !it.IsAtEnd() )
      {
      roiSize[0] = 2*roiRadius+1;
      roiSize[1] = 2*roiRadius+1;;

      if ( it.Get() == 1 )
        {
        index[0] = it.GetIndex()[0];
        index[1] = it.GetIndex()[1];
        index[2] = 0;

        point[0] = static_cast< double >( index[0] )*spacing[0] + origin[0];
        point[1] = static_cast< double >( index[1] )*spacing[1] + origin[1];
        point[2] = 0.0;
        
        (*domainVec).push_back( point );

        roiStart[0] = it.GetIndex()[0] - roiRadius;
        if ( roiStart[0] < 0 )
          {
          roiStart[0] = 0;
          }
        if ( roiStart[0] >= sliceSize[0] )
          {
          roiStart[0] = sliceSize[0]-1;
          }

        roiStart[1] = it.GetIndex()[1] - roiRadius;
        if ( roiStart[1] < 0 )
          {
          roiStart[1] = 0;
          }
        if ( roiStart[1] >= sliceSize[1] )
          {
          roiStart[1] = sliceSize[1]-1;
          }
        roiRegion.SetIndex( roiStart );

        if ( roiStart[0] + roiSize[0] >= sliceSize[0] )
          {
          roiSize[0] = sliceSize[0] - roiStart[0] - 1;
          }
        if ( roiStart[1] + roiSize[1] >= sliceSize[1] )
          {
          roiSize[1] = sliceSize[1] - roiStart[1] - 1;
          }
        roiRegion.SetSize( roiSize );

        SliceIteratorType roiIt( domainSlice, roiRegion );
        
        roiIt.GoToBegin();
        while ( !roiIt.IsAtEnd() )
          {
          if ( roiIt != it )
            {
            if ( roiIt.Get() == 1 )
              {
              roiIt.Set( 0 );
              }
            }
          
          ++roiIt;
          }
        }
      
      ++it;
      }
    }
  while ( (*domainVec).size() > 50 );
}


void GetZValuesFromTPS( std::vector< ImageType::PointType > domainPointsVec, std::vector< double >* rangeValuesVec, 
                        std::vector< ImageType::PointType > surfacePoints, ImageType::Pointer image )
{
  std::vector< double* > surfacePointPtrs;

  for ( unsigned int i=0; i<surfacePoints.size(); i++ )
    {
    double* point = new double[3];

    point[0] = surfacePoints[i][0];
    point[1] = surfacePoints[i][1];
    point[2] = surfacePoints[i][2];

    surfacePointPtrs.push_back( point );
    }

  cipThinPlateSplineSurface tpsSurface( &surfacePointPtrs );

  for ( unsigned int i=0; i<domainPointsVec.size(); i++ )
    {
    double z = tpsSurface.GetSurfaceHeight( (domainPointsVec[i])[0], (domainPointsVec[i])[1] );

    (*rangeValuesVec).push_back( z );
    }
}


PCA GetShapePCA( std::vector< std::vector< double > > valuesVecVec )
{
  PCA pcaModel;

  pcaModel.numModes = 0;

  unsigned int numVecs  = valuesVecVec.size();
  unsigned int numZvals = (valuesVecVec[0]).size();

  //
  // First compute the mean
  //
  for ( unsigned int z=0; z<numZvals; z++ )
    {
    double counter = 0.0;

    for ( unsigned int v=0; v<numVecs; v++ )
      {
      counter += (valuesVecVec[v])[z];
      }

    double meanVal = counter/static_cast< double >( numVecs );

    pcaModel.meanVec.push_back( meanVal );
    }

  //
  // Now construct the matrix X, which is a (numVecs) X (numZvals)
  // matrix, each row of which contains a mean-centered training
  // sample entry 
  //
  vnl_matrix< double > X( numVecs, numZvals );

  for ( unsigned int r=0; r<numVecs; r++ )
    {
    for ( unsigned int c=0; c<numZvals; c++ )
      {
      X[r][c] = (valuesVecVec[r])[c] - pcaModel.meanVec[c];
      }
    }

  //
  // Construct XXt, which is the product of X and the transpose of
  // X. We need this matrix for computing PCA.
  //
  vnl_matrix< double > Xt( numZvals, numVecs );
  Xt = X.transpose();

  vnl_matrix< double > XXt( numVecs, numVecs );
  XXt = X*Xt;

  //
  // Now we need to get the eigenvalues and eigenvectors of XXt
  //
  vnl_symmetric_eigensystem< double >  eig( XXt );

  for ( unsigned int i=0; i<numZvals; i++ )
    {
    pcaModel.modeVec.push_back( 0.0 );

    if ( i<numVecs )
      {
      if ( eig.get_eigenvalue( numVecs-i-1 ) > 0.0 )
        {
        pcaModel.modeVec[i] = eig.get_eigenvalue( numVecs-i-1 );

        pcaModel.numModes++;
        }
      }
    }

  //
  // We need to normalize the modeWeights by the number of non-zero
  // eigenvalues of XXt. This will give us properly scaled eigenvalues
  // within our (numZvals)-D space
  //
  for ( unsigned int i=0; i<pcaModel.numModes; i++ )
    {
    pcaModel.modeVec[i] = pcaModel.modeVec[i]/static_cast< double >( pcaModel.numModes );
    }

  //
  // Finally, compute the eigenvalues in our (numZVals)-D space
  //
  for ( unsigned int v=0; v<pcaModel.numModes; v++ )
    {
    std::vector< double > modeVec;
    for ( unsigned int z=0; z<numZvals; z++ )
      {
      modeVec.push_back( 0.0 );
      }

    if ( pcaModel.modeVec[v] > 0.0 )
      {
      vnl_vector< double > u( numZvals );
      
      u = Xt*eig.get_eigenvector( numVecs-v-1 );

      //
      // Compute u's magnitude
      //
      double mag = vcl_sqrt( dot_product( u, u ) );

      for ( unsigned int z=0; z<numZvals; z++ )
        {
        modeVec[z] = u[z]/mag;
        }
      }

    pcaModel.modeVecVec.push_back( modeVec );
    }

  return pcaModel;
}


#endif
