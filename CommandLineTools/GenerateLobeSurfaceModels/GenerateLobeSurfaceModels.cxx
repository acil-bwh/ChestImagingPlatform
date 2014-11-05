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
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "GenerateLobeSurfaceModelsHelper.h"
#include "GenerateLobeSurfaceModelsCLP.h"
#include "cipMacro.h"
#include "itksys/Directory.hxx"

void GetResourceFiles( std::string, std::vector< std::string >*, std::vector< std::string >* );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::vector< std::string > trainTransformFileVec;
  std::vector< std::string > trainPointsFileVec;

  GetResourceFiles( resourcesDir, &trainTransformFileVec, &trainPointsFileVec );
  cipAssert( trainTransformFileVec.size() == trainPointsFileVec.size() );

  if ( trainPointsFileVec.size() < 2 )
    {
      std::cerr << "Insufficient resources in directory" << std::endl;
      return cip::EXITFAILURE;
    }

  std::cout << "Read in " << trainPointsFileVec.size() << " atlas resource file pairs." << std::endl;

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

  cipAssert( leftWidth[0] > 0.0 );
  cipAssert( leftWidth[1] > 0.0 );

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

  cipAssert( rightWidth[0] > 0.0 );
  cipAssert( rightWidth[1] > 0.0 );

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

    rightRangeValuesVecVec.push_back( rightRangeValuesVec );
    loRangeValuesVecVec.push_back( loRangeValuesVec );
    }

  // Now we have all the training information needed for building the
  // PCA-based fissure models for this case. 
  PCA loShapePCA;
  PCA rightShapePCA;
  if ( loRangeValuesVecVec.size() < loRangeValuesVecVec[0].size() )
    {
      loShapePCA = GetHighDimensionPCA( loRangeValuesVecVec );
    }
  else
    {
      loShapePCA = GetLowDimensionPCA( loRangeValuesVecVec );
    }

  if ( rightRangeValuesVecVec.size() < rightRangeValuesVecVec[0].size() )
    {
      rightShapePCA = GetHighDimensionPCA( rightRangeValuesVecVec );
    }
  else
    {
      rightShapePCA = GetLowDimensionPCA( rightRangeValuesVecVec );
    }

  // Now create shape models for each of the three lobe boundaries. To do this we need to 
  // collect the domain locations and mean vector into a single collection of points
  // for each of the three boundaries
  std::vector< cip::PointType > rightMeanSurfacePoints;
  for ( unsigned int i=0; i<rightShapePCA.meanVec.size(); i++ )
    {
      unsigned int index = i%(rightShapePCA.meanVec.size()/2);

      cip::PointType point(3);
        point[0] = rightDomainPatternPoints[index][0];
  	point[1] = rightDomainPatternPoints[index][1];
  	point[2] = rightShapePCA.meanVec[i];

      rightMeanSurfacePoints.push_back( point );
    }

  std::vector< cip::PointType > loMeanSurfacePoints;
  for ( unsigned int i=0; i<loShapePCA.meanVec.size(); i++ )
    {
      cip::PointType point(3);
        point[0] = leftDomainPatternPoints[i][0];
  	point[1] = leftDomainPatternPoints[i][1];
  	point[2] = loShapePCA.meanVec[i];

      loMeanSurfacePoints.push_back( point );
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

void GetResourceFiles( std::string resourcesDir, std::vector< std::string >* trainTransformFileVec, 
		       std::vector< std::string >* trainPointsFileVec)
{
  itksys::Directory dir;
  dir.Load( resourcesDir.c_str() );

  unsigned int numFiles = itksys::Directory::GetNumberOfFilesInDirectory( resourcesDir.c_str() );
  std::string transformFileSuffix = "_TO_10002K_INSP_STD_BWH_COPD_transform.tfm";
  std::string pointsFileSuffix    = "_regionAndTypePoints.csv";

  for ( unsigned int i=0; i<numFiles; i++ )
    {
      std::size_t pointsFileSuffixStart = std::string(dir.GetFile( i )).find( pointsFileSuffix );
      if ( pointsFileSuffixStart != std::string::npos )
	{
	  std::string pointsFilePrefix = std::string(dir.GetFile( i )).substr(0, pointsFileSuffixStart);
	  for ( unsigned int j=0; j<numFiles; j++ )
	    {
	      std::size_t transformFileSuffixStart = std::string(dir.GetFile( j )).find( transformFileSuffix );
	      if ( transformFileSuffixStart != std::string::npos )
		{
		  std::string transformFilePrefix = std::string(dir.GetFile( j )).substr(0, transformFileSuffixStart);
		  if ( transformFilePrefix.compare( pointsFilePrefix ) == 0 )
		    {
		      trainPointsFileVec->push_back( resourcesDir + std::string(dir.GetFile(i)) );
		      trainTransformFileVec->push_back( resourcesDir + std::string(dir.GetFile(j)) );
		    }
		}
	    }
	}
    }
}

#endif
