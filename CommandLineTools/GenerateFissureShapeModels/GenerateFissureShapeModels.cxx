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

#include <tclap/CmdLine.h>
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
#include "cipLobeBoundaryShapeModel.h"
#include "cipLobeBoundaryShapeModelIO.h"


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
  //
  // Eigenvalues are in descending order
  //
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


int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::vector< std::string > trainTransformFileVec;
  std::vector< std::string > trainPointsFileVec;

  std::string inputFileName                    = "NA";
  std::string refImageFileName                 = "NA";
  std::string refPointsFileName                = "NA";
  std::string rhShapeModelFileName             = "NA";
  std::string roShapeModelFileName             = "NA";
  std::string loShapeModelFileName             = "NA";
  std::string outRefToInputTransformFileName   = "NA";
  float maxStepLength                          = 1.0;
  float minStepLength                          = 0.001;
  float translationScale                       = 0.001;
  float downsampleFactor                       = 4.0;
  unsigned int numberOfIterations               = 20;

  //
  // Argument descriptions for user 
  //
  std::string programDescription = "This program is used to generate \
fissure shape models based \
on a training set. The basic approach is as follows: a reference image \
is read and then registered to the label map for which the user \
wants to produce fissure models. Additionally, a set of training \
data is read in. These data sets consist of fissure indices and a \
transform file that indicates the mapping between the training data \
label map and the reference label map. Once the reference image is \
registered to the input image, the resulting transform in \
conjunction with the training data transform is used to map the \
training fissure points to the input image's coordinate frame. This \
is done repeatedly for all the training data sets. Once all the \
fissure points are in the coordinate frame of the input image, PCA \
is performed to get the mean and modes of the fissure points. This \
data is then printed to file. \
\
The reference image for model building has been (arbitrarily) \
chosen to be the COPDGene case, 10002K_INSP_STD_BWH_COPD. There are \
20 training datasets consisting, including INSP and EXP data. These \
datasets were manually segmented by George Washko and Alejandro Diaz \
for the MICCAI 09 submission. The transform files used to map the \
training data to the reference dataset and the files for the training \
data can be found in ~/Processed/Atlases/LungLobeAtlases \
\
The output file format is as follows: the first line contains the \
origin of the input image. The second line contains the spacing of \
the input image. The third line indicates the number of modes, \
N. The fourth line is the mean vector. The entries are for the \
continuous index z-values of the control points. The next N lines \
contain the mode variances and a weight value. Upon generation of \
the shape model file, the weight value will be 0. Later in the lobe \
segmentation process, this value can be overwritten with a weight \
has been found by fitting to particles data. The next N lines are \
the modes themselves -- again, the modes represent the variation of \
the control points in the z-direction (the range of our TPS \
function). The length of each mode (and mean) vector is m, where m \
indicates the number of control points. Finally, the next m lines \
indicate the control point indices (the domain locations). Note \
that the z-value for these entries is 0. This is to emphasize that \
they are in the domain -- the x-y plane.";

  std::string trainTransformFileVecDesc           = "Training data set transform file. This file contains the \
transform that is used to map the corresponding training data set's fissure points to the reference image's \
coordinate system. It should immediately be followed with a region and type points file (specified with the \
--trainPoints flag)";
  std::string trainPointsFileVecDesc              = "Region and type points file corresponding to the previously \
called transform file (see notes for --trainTrans)";
  std::string inputFileNameDesc                   = "Input label map to serve as starting point for lung lobe \
segmentation .The level of lung segmentation 'grunularity' should be at the left lung right lung split \
level. In other words, a mask divided into thirds (left and right) will work fine, as will one for which \
only the left and right lung are labeled.";
  std::string refImageFileNameDesc                = "Reference image label map corresponding to the reference points. \
This is the image to which all other image data in the training set is registered to";
  std::string refPointsFileNameDesc               = "Reference points corresponding to the reference image.";
  std::string rhShapeModelFileNameDesc            = "Output right horizontal shape model file name";
  std::string roShapeModelFileNameDesc            = "Output right oblique shape model file name";
  std::string loShapeModelFileNameDesc            = "Output left oblique shape model file name";
  std::string outRefToInputTransformFileNameDesc  = "Output reference image to input label map transform file name";
  std::string maxStepLengthDesc                   = "Max step length. Default is 1.0";
  std::string minStepLengthDesc                   = "Min step length. Default is 0.001";   
  std::string numberOfIterationsDesc              = "Number of iterations. Default is 20";     
  std::string translationScaleDesc                = "Translation scale. Default is 0.001";
  std::string downsampleFactorDesc                = "Down sample factor. The fixed and moving images will be \
down sampled by this amount before registration. (Default is 1.0, i.e. no down sampling).";

  unsigned int numTrainingSets = 0;

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 251 $" );

    TCLAP::MultiArg<std::string> trainTransformFileVecArg( "", "trainTrans", trainTransformFileVecDesc, true, "string", cl );
    TCLAP::MultiArg<std::string> trainPointsFileVecArg( "", "trainPoints", trainPointsFileVecDesc, true, "string", cl );
    TCLAP::ValueArg<std::string> inputFileNameArg( "i", "inFileName", inputFileNameDesc, true, inputFileName, "string", cl );
    TCLAP::ValueArg<std::string> refImageFileNameArg( "", "refim", refImageFileNameDesc, true, refImageFileName, "string", cl );
    TCLAP::ValueArg<std::string> refPointsFileNameArg( "", "refps", refPointsFileNameDesc, true, refPointsFileName, "string", cl );
    TCLAP::ValueArg<std::string> rhShapeModelFileNameArg( "", "orh", rhShapeModelFileNameDesc, false, rhShapeModelFileName, "string", cl );
    TCLAP::ValueArg<std::string> roShapeModelFileNameArg( "", "oro", roShapeModelFileNameDesc, false, roShapeModelFileName, "string", cl );
    TCLAP::ValueArg<std::string> loShapeModelFileNameArg( "", "olo", loShapeModelFileNameDesc, false, loShapeModelFileName, "string", cl );
    TCLAP::ValueArg<std::string> outRefToInputTransformFileNameArg( "", "ot", outRefToInputTransformFileNameDesc, false, outRefToInputTransformFileName, "string", cl );
    TCLAP::ValueArg<float> maxStepLengthArg( "", "max", maxStepLengthDesc, false, maxStepLength, "float", cl );
    TCLAP::ValueArg<float> minStepLengthArg( "", "min", minStepLengthDesc, false, minStepLength, "float", cl );
    TCLAP::ValueArg<float> translationScaleArg( "t", "transScale", translationScaleDesc, false, translationScale, "float", cl );
    TCLAP::ValueArg<float> downsampleFactorArg( "d", "down", downsampleFactorDesc, false, downsampleFactor, "float", cl );
    TCLAP::ValueArg<unsigned int> numberOfIterationsArg( "n", "numIters", numberOfIterationsDesc, false, numberOfIterations, "unsigned int", cl );

    cl.parse( argc, argv );

    for ( unsigned int i=0; i<trainTransformFileVecArg.getValue().size(); i++ )
      {
	trainTransformFileVec.push_back( trainTransformFileVecArg.getValue()[i] );
      }
    for ( unsigned int i=0; i<trainPointsFileVecArg.getValue().size(); i++ )
      {
	trainPointsFileVec.push_back( trainPointsFileVecArg.getValue()[i] );
      }

    numTrainingSets                = trainPointsFileVec.size();

    inputFileName                  = inputFileNameArg.getValue();
    refImageFileName               = refImageFileNameArg.getValue();
    refPointsFileName              = refPointsFileNameArg.getValue();
    rhShapeModelFileName           = rhShapeModelFileNameArg.getValue();
    roShapeModelFileName           = roShapeModelFileNameArg.getValue();
    loShapeModelFileName           = loShapeModelFileNameArg.getValue();
    outRefToInputTransformFileName = outRefToInputTransformFileNameArg.getValue();
    maxStepLength                  = maxStepLengthArg.getValue();
    minStepLength                  = minStepLengthArg.getValue();
    translationScale               = translationScaleArg.getValue();
    downsampleFactor               = downsampleFactorArg.getValue();
    numberOfIterations             = numberOfIterationsArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

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

  TransformType::Pointer refToInputTransform = TransformType::New();

  //
  // We need to get the transform that maps the reference image (from
  // our training set) to the input image (whose lobes we want to
  // segment). We read both these images, downsample them, and then
  // register them. Once this has been done, we can use the resulting
  // transform to map all training points to the input image's
  // coordinate frame
  //
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

  //
  // Now extract the whole lung region from the resized fixed and
  // moving images
  //
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

  //
  // Now that we have the transform that maps the reference image to
  // the input image, we need to read in and map all the training
  // points so that they are in the input image's coordinate frame.
  // The following three vectors of vectors will contain all the
  // points from the training set (transformed to be in the input
  // image's coordinate frame)
  //
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

  //
  // At this stage all the training points should be registered to the
  // input image's coordinate frame. We can now get the domain
  // locations for each of the three fissures. Note that the domain
  // locations have the z-coordinate zeroed out.
  //
  std::cout << "Getting domain locations..." << std::endl;
  std::vector< ImageType::PointType > loDomainVec;
  GetDomainPoints( loTransPointsVecVec, &loDomainVec, fixedReader->GetOutput(), 0 );

  std::vector< ImageType::PointType > roDomainVec;
  GetDomainPoints( roTransPointsVecVec, &roDomainVec, fixedReader->GetOutput(), 1 );

  std::vector< ImageType::PointType > rhDomainVec;
  GetDomainPoints( rhTransPointsVecVec, &rhDomainVec, fixedReader->GetOutput(), 2 );

  //
  // Now we have the domain locations for each of our three
  // fissures. To build our PCA model, we need to evaluate the z
  // values at the domain locations for each of our training
  // datasets. 
  //
  std::cout << "Getting range locations..." << std::endl;
  std::vector< std::vector< double > > rhRangeValuesVecVec;
  std::vector< std::vector< double > > roRangeValuesVecVec;
  std::vector< std::vector< double > > loRangeValuesVecVec;

  for ( unsigned int i=0; i<numTrainingSets; i++ )
    {
    std::vector< double > roRangeValuesVec;
    std::vector< double > rhRangeValuesVec;
    std::vector< double > loRangeValuesVec;

    GetZValuesFromTPS( roDomainVec, &roRangeValuesVec, roTransPointsVecVec[i], fixedReader->GetOutput() );
    GetZValuesFromTPS( rhDomainVec, &rhRangeValuesVec, rhTransPointsVecVec[i], fixedReader->GetOutput() );
    GetZValuesFromTPS( loDomainVec, &loRangeValuesVec, loTransPointsVecVec[i], fixedReader->GetOutput() );

    roRangeValuesVecVec.push_back( roRangeValuesVec );
    rhRangeValuesVecVec.push_back( rhRangeValuesVec );
    loRangeValuesVecVec.push_back( loRangeValuesVec );
    }

  //
  // Now we have all the training information needed for building the
  // PCA-based fissure models for this case. 
  //
  PCA rhShapePCA = GetShapePCA( rhRangeValuesVecVec );
  PCA roShapePCA = GetShapePCA( roRangeValuesVecVec );
  PCA loShapePCA = GetShapePCA( loRangeValuesVecVec );

  //
  // Now create shape models for each of the three lobe boundaries. To do this we need to 
  // collect the domain locations and mean vector into a single collection of points
  // for each of the three boundaries
  //
  std::vector< double* >* roMeanSurfacePoints = new std::vector< double* >;
  for ( unsigned int i=0; i<roShapePCA.meanVec.size(); i++ )
    {
      double* point = new double[3];
        point[0] = roDomainVec[i][0];
	point[1] = roDomainVec[i][1];
	point[2] = roShapePCA.meanVec[i];

      roMeanSurfacePoints->push_back( point );
    }

  std::vector< double* >* rhMeanSurfacePoints = new std::vector< double* >;
  for ( unsigned int i=0; i<rhShapePCA.meanVec.size(); i++ )
    {
      double* point = new double[3];
        point[0] = rhDomainVec[i][0];
	point[1] = rhDomainVec[i][1];
	point[2] = rhShapePCA.meanVec[i];

      rhMeanSurfacePoints->push_back( point );
    }

  std::vector< double* >* loMeanSurfacePoints = new std::vector< double* >;
  for ( unsigned int i=0; i<loShapePCA.meanVec.size(); i++ )
    {
      double* point = new double[3];
        point[0] = loDomainVec[i][0];
	point[1] = loDomainVec[i][1];
	point[2] = loShapePCA.meanVec[i];

      loMeanSurfacePoints->push_back( point );
    }

  //
  // Now create the boundary models
  //
  std::vector< double > rhModeWeights;
  for ( unsigned int n=0; n<rhShapePCA.modeVec.size(); n++ )
    {
      rhModeWeights.push_back( 0.0 );
    }

  std::vector< double > roModeWeights;
  for ( unsigned int n=0; n<roShapePCA.modeVec.size(); n++ )
    {
      roModeWeights.push_back( 0.0 );
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

  cipLobeBoundaryShapeModel* roShapeModel = new cipLobeBoundaryShapeModel();
    roShapeModel->SetImageOrigin( origin );
    roShapeModel->SetImageSpacing( spacing );
    roShapeModel->SetMeanSurfacePoints( roMeanSurfacePoints );
    roShapeModel->SetEigenvalues( &roShapePCA.modeVec );
    roShapeModel->SetModeWeights( &roModeWeights );
    roShapeModel->SetEigenvectors( &roShapePCA.modeVecVec );
    roShapeModel->SetNumberOfModes( roShapePCA.numModes );

  cipLobeBoundaryShapeModel* rhShapeModel = new cipLobeBoundaryShapeModel();
    rhShapeModel->SetImageOrigin( origin );
    rhShapeModel->SetImageSpacing( spacing );
    rhShapeModel->SetMeanSurfacePoints( rhMeanSurfacePoints );
    rhShapeModel->SetEigenvalues( &rhShapePCA.modeVec );
    rhShapeModel->SetModeWeights( &rhModeWeights );
    rhShapeModel->SetEigenvectors( &rhShapePCA.modeVecVec );
    rhShapeModel->SetNumberOfModes( rhShapePCA.numModes );

  cipLobeBoundaryShapeModel* loShapeModel = new cipLobeBoundaryShapeModel();
    loShapeModel->SetImageOrigin( origin );
    loShapeModel->SetImageSpacing( spacing );
    loShapeModel->SetMeanSurfacePoints( loMeanSurfacePoints );
    loShapeModel->SetModeWeights( &loModeWeights );
    loShapeModel->SetEigenvalues( &loShapePCA.modeVec );
    loShapeModel->SetEigenvectors( &loShapePCA.modeVecVec );
    loShapeModel->SetNumberOfModes( loShapePCA.numModes );

  if ( rhShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right horizontal shape model to file..." << std::endl;
    cipLobeBoundaryShapeModelIO rhWriter;
      rhWriter.SetFileName( rhShapeModelFileName );
      rhWriter.SetInput( rhShapeModel );
      rhWriter.Write();
    }
  if ( roShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing right oblique shape model to file..." << std::endl;
    cipLobeBoundaryShapeModelIO roWriter;
      roWriter.SetFileName( roShapeModelFileName );
      roWriter.SetInput( roShapeModel );
      roWriter.Write();
    }
  if ( loShapeModelFileName.compare( "NA" ) != 0 )
    {
    std::cout << "Writing left oblique shape model to file..." << std::endl;
    cipLobeBoundaryShapeModelIO loWriter;
      loWriter.SetFileName( loShapeModelFileName );
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
  ChestConventions conventions;
  
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
