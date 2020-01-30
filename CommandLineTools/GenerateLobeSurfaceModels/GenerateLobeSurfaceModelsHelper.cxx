#include "GenerateLobeSurfaceModelsHelper.h"

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

      cip::PointType point(3);
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
	      (*leftObliquePointsVec).push_back( itkPoint ); 
	    }
	}
      else if ( cipRegion == static_cast< unsigned char >( cip::RIGHTLUNG ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTSUPERIORLOBE ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTMIDDLELOBE ) ||
		cipRegion == static_cast< unsigned char >( cip::RIGHTINFERIORLOBE ) )
	{
	  if ( cipType == static_cast< unsigned char >( cip::OBLIQUEFISSURE ) )
	    {
	      (*rightObliquePointsVec).push_back( itkPoint ); 
	    }
	  else if ( cipType == static_cast< unsigned char >( cip::HORIZONTALFISSURE ) )
	    {
	      (*rightHorizontalPointsVec).push_back( itkPoint ); 
	    }
	}
    }  
}

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
  std::vector< cip::PointType > surfacePointPtrs;

  for ( unsigned int i=0; i<surfacePoints.size(); i++ )
    {
      cip::PointType point(3);
        point[0] = surfacePoints[i][0];
	point[1] = surfacePoints[i][1];
	point[2] = surfacePoints[i][2];
      
      surfacePointPtrs.push_back( point );
    }

  cipThinPlateSplineSurface tpsSurface( surfacePointPtrs );

  for ( unsigned int i=0; i<domainPointsVec.size(); i++ )
    {
    double z = tpsSurface.GetSurfaceHeight( (domainPointsVec[i])[0], (domainPointsVec[i])[1] );

    (*rangeValuesVec).push_back( z );
    }
}

PCA GetLowDimensionPCA( std::vector< std::vector< double > > valuesVecVec )
{
  PCA pcaModel;

  pcaModel.numModes = 0;

  unsigned int numSamples  = valuesVecVec.size();
  unsigned int dimension = (valuesVecVec[0]).size();

  // First compute the mean
  for ( unsigned int d=0; d<dimension; d++ )
    {
    double counter = 0.0;

    for ( unsigned int s=0; s<numSamples; s++ )
      {
      counter += (valuesVecVec[s])[d];
      }

    double meanVal = counter/double( numSamples );

    pcaModel.meanVec.push_back( meanVal );
    }

  // Now construct the covariance S, which is a (dimension) X (dimension)
  // matrix, each row of which contains a mean-centered training
  // sample entry 
  vnl_matrix< double > S( dimension, dimension );

  for ( unsigned int r=0; r<dimension; r++ )
    {
      for ( unsigned int c=0; c<dimension; c++ )
	{
	  S[r][c] = 0.0;
	  for ( unsigned int d=0; d<numSamples; d++ )
	    {	      
	      S[r][c] += (1/double(numSamples))*((valuesVecVec[d])[r] - pcaModel.meanVec[r])*
		((valuesVecVec[d])[c] - pcaModel.meanVec[c]);
	    }
	}
    }

  // Now we need to get the eigenvalues and eigenvectors of S
  vnl_symmetric_eigensystem< double >  eig( S );
  
  for ( unsigned int i=0; i<dimension; i++ )
    {
      pcaModel.modeVec.push_back( 0.0 );
      pcaModel.modeVec[i] = eig.get_eigenvalue( dimension-i-1 );
      
      pcaModel.numModes++;
    }
 
  // Get the eigenvectors
  for ( unsigned int v=0; v<pcaModel.numModes; v++ )
    {
      std::vector< double > modeVec;
      for ( unsigned int d=0; d<dimension; d++ )
	{
	  modeVec.push_back( 0.0 );
	}
      
      for ( unsigned int d=0; d<dimension; d++ )
        {
	  modeVec[d] = eig.get_eigenvector( dimension-v-1 )[d];
        }
      
      pcaModel.modeVecVec.push_back( modeVec );
    }
  
  return pcaModel;   
}

PCA GetHighDimensionPCA( std::vector< std::vector< double > > valuesVecVec )
{
  PCA pcaModel;

  pcaModel.numModes = 0;

  unsigned int numVecs  = valuesVecVec.size();
  unsigned int numZvals = (valuesVecVec[0]).size();

  // First compute the mean
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

  // Now construct the matrix X, which is a (numVecs) X (numZvals)
  // matrix, each row of which contains a mean-centered training
  // sample entry 
  vnl_matrix< double > X( numVecs, numZvals );

  for ( unsigned int r=0; r<numVecs; r++ )
    {
    for ( unsigned int c=0; c<numZvals; c++ )
      {
      X[r][c] = (valuesVecVec[r])[c] - pcaModel.meanVec[c];
      }
    }

  // Construct XXt, which is the product of X and the transpose of
  // X. We need this matrix for computing PCA.
  vnl_matrix< double > Xt( numZvals, numVecs );
  Xt = X.transpose();

  vnl_matrix< double > XXt( numVecs, numVecs );
  XXt = X*Xt;

  // Now we need to get the eigenvalues and eigenvectors of XXt
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

  // We need to normalize the modeWeights by the number of non-zero
  // eigenvalues of XXt. This will give us properly scaled eigenvalues
  // within our (numZvals)-D space
  for ( unsigned int i=0; i<pcaModel.numModes; i++ )
    {
    pcaModel.modeVec[i] = pcaModel.modeVec[i]/static_cast< double >( pcaModel.numModes );
    }

  // Finally, compute the eigenvalues in our (numZVals)-D space
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

      // Compute u's magnitude
      double mag = std::sqrt( dot_product( u, u ) );

      for ( unsigned int z=0; z<numZvals; z++ )
        {
        modeVec[z] = u[z]/mag;
        }
      }

    pcaModel.modeVecVec.push_back( modeVec );
    }

  return pcaModel;
}
