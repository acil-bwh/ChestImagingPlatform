#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <fstream>
#include <cmath>
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkPolyData.h"
#include "vtkFloatArray.h"
#include "vtkPolyDataReader.h"
#include "vtkPointData.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "cipExceptionObject.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkDiscreteHessianGaussianImageFunction.h"
#include "itkDiscreteGaussianDerivativeImageFunction.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipVesselParticleConnectedComponentFilter.h"
#include "ComputeFissureFeatureVectorsCLP.h"

typedef itk::Image< unsigned char, 3 >                                            MaskType;
typedef itk::Image< float, 3 >                                                    FloatImageType;
typedef itk::Image< float, 3 >                                                    DistanceImageType;
typedef itk::ImageFileWriter< DistanceImageType >                                 DistanceWriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< MaskType, DistanceImageType >    DistanceMapType;
typedef itk::ImageRegionIteratorWithIndex< cip::CTType >                          CTIteratorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                    LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< DistanceImageType >                    DistanceImageIteratorType;
typedef itk::DiscreteHessianGaussianImageFunction< cip::CTType >                  HessianImageFunctionType ;
typedef itk::DiscreteGaussianDerivativeImageFunction< cip::CTType >               DerivativeFunctionType;

struct FEATUREVECTOR
{
  double eigenVector[3];
  short  intensity;
  double distanceToVessel;
  double distanceToLobeSurface;
  double angleWithLobeSurfaceNormal;
  double pMeasure;
  double fMeasure;
  double gradX;
  double gradY;
  double gradZ;
  double gradientMagnitude;
  std::list< double > gradient;
  std::list< double > eigenValues;
  std::list< double > eigenValueMags;
};

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType, cip::CTType::SizeType, 
						 cip::CTType::PointType, vtkSmartPointer< vtkPolyData > );
FEATUREVECTOR ComputeFissureFeatureVector( vtkSmartPointer< vtkPolyData >, cip::CTType::Pointer, 
					   DistanceImageType::Pointer, const cipThinPlateSplineSurface& ,  
					   const cipThinPlateSplineSurface&,  const cipThinPlateSplineSurface&,
					   DerivativeFunctionType::Pointer, HessianImageFunctionType::Pointer, 
					   cip::CTType::IndexType );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instatiate ChestConventions for general convenience later
  cip::ChestConventions conventions;

  if ( pointsParticlesFileName.compare( "NA" ) == 0 )
    {
      std::cerr << "Must specify a points particles file." << std::endl;
      return 1;
    }

  cipThinPlateSplineSurface rhTPS;
  cipThinPlateSplineSurface roTPS;
  cipThinPlateSplineSurface loTPS;

  if ( rightShapeModelFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading right shape model..." << std::endl;
      cip::LobeSurfaceModelIO rightShapeModelIO;
        rightShapeModelIO.SetFileName( rightShapeModelFileName );
      try
  	{
  	rightShapeModelIO.Read();
  	}
      catch ( cip::ExceptionObject &excp )
  	{
  	std::cerr << "Exception caught reading right shape model:";
  	std::cerr << excp << std::endl;
  	return cip::EXITFAILURE;
  	}
      rightShapeModelIO.GetOutput()->SetRightLungSurfaceModel( true );

      rhTPS.SetSurfacePoints( rightShapeModelIO.GetOutput()->GetMeanRightHorizontalSurfacePoints() );
      roTPS.SetSurfacePoints( rightShapeModelIO.GetOutput()->GetMeanRightObliqueSurfacePoints() );
    }
  else if ( leftShapeModelFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left shape model..." << std::endl;
      cip::LobeSurfaceModelIO leftShapeModelIO;
        leftShapeModelIO.SetFileName( leftShapeModelFileName );
      try
  	{
  	leftShapeModelIO.Read();
  	}
      catch ( cip::ExceptionObject &excp )
  	{
  	std::cerr << "Exception caught reading left shape model:";
  	std::cerr << excp << std::endl;
  	return cip::EXITFAILURE;
  	}

      loTPS.SetSurfacePoints( leftShapeModelIO.GetOutput()->GetMeanSurfacePoints() );
    }
  else 
    {
      std::cerr << "Must specify a shape model file name." << std::endl;
      return 1;
    }

  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer lmReader = cip::LabelMapReaderType::New();
    lmReader->SetFileName( lmFileName );
  try
    {
    lmReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading label map:";
    std::cerr << excp << std::endl;
    }

  std::cout << "Reading CT image..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while reading label map:";
    std::cerr << excp << std::endl;      
    return cip::NRRDREADFAILURE;
    }

  std::cout << "Reading vessel particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > vesselParticlesReader = vtkPolyDataReader::New();
    vesselParticlesReader->SetFileName( vesselParticlesFileName.c_str() );
    vesselParticlesReader->Update();    

  vtkSmartPointer< vtkPolyData > vesselParticles = vtkSmartPointer< vtkPolyData >::New();
  cip::TransferFieldDataToFromPointData( vesselParticlesReader->GetOutput(), vesselParticles, 
  					 true, false, true, false );

  // Input vessel particles are expected to be "raw" (unfiltered). We filter here
  // for consistency
  double interParticleSpacing = 1.5;
  unsigned int componentSizeThreshold = 10;
  double maxAllowableDistance = 3.0; 
  double particleAngleThreshold = 20.0;
  double scaleRatioThreshold = 0.25;
  unsigned int maxComponentSize = std::numeric_limits<unsigned int>::max();
  double maxAllowableScale = 5.0;
  double minAllowableScale = 0.0;

  std::cout << "Filtering vessel particles..." << std::endl;
  cipVesselParticleConnectedComponentFilter* filter = new cipVesselParticleConnectedComponentFilter();
    filter->SetComponentSizeThreshold( componentSizeThreshold );
    filter->SetParticleDistanceThreshold( maxAllowableDistance );
    filter->SetParticleAngleThreshold( particleAngleThreshold );
    filter->SetScaleRatioThreshold( scaleRatioThreshold );
    filter->SetMaximumComponentSize( maxComponentSize );
    filter->SetMaximumAllowableScale( maxAllowableScale );
    filter->SetMinimumAllowableScale( minAllowableScale );
    filter->SetInput( vesselParticles );
    filter->Update();

  if ( filter->GetOutput()->GetNumberOfPoints() == 0 )
    {
      std::cerr << "No vessel particles. Exiting." << std::endl;
      return cip::EXITFAILURE;
    }

  std::cout << "Getting vessel distance map..." << std::endl;
  DistanceImageType::Pointer vesselDistanceMap = DistanceImageType::New();
  vesselDistanceMap = 
    GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
  			  ctReader->GetOutput()->GetOrigin(), filter->GetOutput() );

  std::cout << "Reading particles file indicating where to compute feature vectors..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > pointsParticlesReader = vtkPolyDataReader::New();
    pointsParticlesReader->SetFileName( pointsParticlesFileName.c_str() );
    pointsParticlesReader->Update();    

  std::cout << "Getting points particles distance map..." << std::endl;
  DistanceImageType::Pointer pointsParticlesDistanceMap = DistanceImageType::New();
  pointsParticlesDistanceMap = 
    GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
  			  ctReader->GetOutput()->GetOrigin(), pointsParticlesReader->GetOutput() );

  unsigned int maxKernelWidth = 100;
  double variance = 1.0;
  double maxError = 0.01;

  HessianImageFunctionType::Pointer hessianFunction = HessianImageFunctionType::New();
    hessianFunction->SetUseImageSpacing( true );
    hessianFunction->SetNormalizeAcrossScale( false );
    hessianFunction->SetInputImage( ctReader->GetOutput() );
    hessianFunction->SetMaximumError( maxError );
    hessianFunction->SetMaximumKernelWidth( maxKernelWidth );
    hessianFunction->SetVariance( variance );
    hessianFunction->Initialize();

  DerivativeFunctionType::Pointer derivativeFunction = DerivativeFunctionType::New();
    derivativeFunction->SetInputImage( ctReader->GetOutput() );
    derivativeFunction->SetUseImageSpacing( true );
    derivativeFunction->SetNormalizeAcrossScale( false );
    derivativeFunction->SetMaximumError( maxError );
    derivativeFunction->SetMaximumKernelWidth( maxKernelWidth );
    derivativeFunction->SetVariance( variance );

  // Now loop through the query points and compute the feature vectors for the true 
  // fissure particles
  std::vector< FEATUREVECTOR > trueFeatureVectors;
  std::cout << "Computing feature vectors..." << std::endl;
  cip::CTType::IndexType index;
  cip::CTType::PointType point;

  for ( unsigned int i=0; i<pointsParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
      point[0] = pointsParticlesReader->GetOutput()->GetPoint(i)[0];
      point[1] = pointsParticlesReader->GetOutput()->GetPoint(i)[1];
      point[2] = pointsParticlesReader->GetOutput()->GetPoint(i)[2];

      ctReader->GetOutput()->TransformPhysicalPointToIndex( point, index );

      FEATUREVECTOR vec = ComputeFissureFeatureVector( pointsParticlesReader->GetOutput(), ctReader->GetOutput(), 
  						       vesselDistanceMap, rhTPS, roTPS, loTPS, derivativeFunction, 
  						       hessianFunction, index );
      if ( *vec.eigenValues.begin() < 0 )
  	{
  	  trueFeatureVectors.push_back( vec );
  	}
    }

  // Now loop through the image and get feature vectors for false examples
  std::vector< FEATUREVECTOR > falseFeatureVectors;

  CTIteratorType cIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType lIt( lmReader->GetOutput(), lmReader->GetOutput()->GetBufferedRegion() );
  DistanceImageIteratorType dIt( pointsParticlesDistanceMap, pointsParticlesDistanceMap->GetBufferedRegion() );

  cIt.GoToBegin();
  lIt.GoToBegin();
  dIt.GoToBegin();
  while ( !cIt.IsAtEnd() )
    {
      if ( lIt.Get() > 0 && cIt.Get() <= -650 )
  	{
  	  unsigned char cipRegion = conventions.GetChestRegionFromValue( lIt.Get() );
  	  if ( (conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::LEFTLUNG) ) &&
  		loTPS.GetNumberSurfacePoints() > 0) || 
  	       (conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::RIGHTLUNG) ) &&
  		roTPS.GetNumberSurfacePoints() > 0 && rhTPS.GetNumberSurfacePoints() > 0) )
  	    {
  	      if ( rand() % 10000 < 3 && std::abs(dIt.Get()) > 2 )
  		{
  		  FEATUREVECTOR vec = ComputeFissureFeatureVector( pointsParticlesReader->GetOutput(), ctReader->GetOutput(), 
  								   vesselDistanceMap, rhTPS, roTPS, loTPS, derivativeFunction, 
  								   hessianFunction, cIt.GetIndex() );
  		  if ( *vec.eigenValues.begin() < 0 )
  		    {
  		      falseFeatureVectors.push_back( vec );
  		    }
  		}
  	    } 
  	}

      ++cIt;
      ++lIt;
      ++dIt;
    }

  std::cout << "Writing true feature vectors to file..." << std::endl;
  std::ofstream trueFile( trueOutFileName.c_str() );

  trueFile << "eigenValue0,eigenValue1,eigenValue2,eigenValueMag0,eigenValueMag1,eigenValueMag2,intensity,";
  trueFile << "distanceToVessel,distanceToLobeSurface,angleWithLobeSurfaceNormal,pMeasure,fMeasure,";
  trueFile << "gradX,gradY,gradZ,gradientMagnitude,gradMin,gradMid,gradMax" << std::endl;
  for ( unsigned int i=0; i<trueFeatureVectors.size(); i++ )
    {
      std::list<double>::iterator itv = trueFeatureVectors[i].eigenValues.begin();
      std::list<double>::iterator itm = trueFeatureVectors[i].eigenValueMags.begin();
      std::list<double>::iterator itg = trueFeatureVectors[i].gradient.begin();
      trueFile << *itv << ","; ++itv;
      trueFile << *itv << ","; ++itv;
      trueFile << *itv << ","; 
      trueFile << *itm << ","; ++itm;
      trueFile << *itm << ","; ++itm;
      trueFile << *itm << ","; 
      trueFile << trueFeatureVectors[i].intensity << ",";
      trueFile << trueFeatureVectors[i].distanceToVessel << ",";
      trueFile << trueFeatureVectors[i].distanceToLobeSurface << ",";
      trueFile << trueFeatureVectors[i].angleWithLobeSurfaceNormal << ",";
      trueFile << trueFeatureVectors[i].pMeasure << ",";
      trueFile << trueFeatureVectors[i].fMeasure << ",";      
      trueFile << trueFeatureVectors[i].gradX << ",";
      trueFile << trueFeatureVectors[i].gradY << ",";
      trueFile << trueFeatureVectors[i].gradZ << ",";
      trueFile << trueFeatureVectors[i].gradientMagnitude << ",";
      trueFile << *itg << ","; ++itg;
      trueFile << *itg << ","; ++itg;
      trueFile << *itg << std::endl;
    }
  trueFile.close();

  std::cout << "Writing false feature vectors to file..." << std::endl;
  std::ofstream falseFile( falseOutFileName.c_str() );

  falseFile << "eigenValue0,eigenValue1,eigenValue2,eigenValueMag0,eigenValueMag1,eigenValueMag2,intensity,";
  falseFile << "distanceToVessel,distanceToLobeSurface,angleWithLobeSurfaceNormal,pMeasure,fMeasure,";
  falseFile << "gradX,gradY,gradZ,gradientMagnitude,gradMin,gradMid,gradMax" << std::endl;
  for ( unsigned int i=0; i<falseFeatureVectors.size(); i++ )
    {
      std::list<double>::iterator itv = falseFeatureVectors[i].eigenValues.begin();
      std::list<double>::iterator itm = falseFeatureVectors[i].eigenValueMags.begin();
      std::list<double>::iterator itg = falseFeatureVectors[i].gradient.begin();
      falseFile << *itv << ","; ++itv;
      falseFile << *itv << ","; ++itv;
      falseFile << *itv << ","; 
      falseFile << *itm << ","; ++itm;
      falseFile << *itm << ","; ++itm;
      falseFile << *itm << ","; 
      falseFile << falseFeatureVectors[i].intensity << ",";
      falseFile << falseFeatureVectors[i].distanceToVessel << ",";
      falseFile << falseFeatureVectors[i].distanceToLobeSurface << ",";
      falseFile << falseFeatureVectors[i].angleWithLobeSurfaceNormal << ",";
      falseFile << falseFeatureVectors[i].pMeasure << ",";
      falseFile << falseFeatureVectors[i].fMeasure << ",";
      falseFile << falseFeatureVectors[i].gradX << ",";
      falseFile << falseFeatureVectors[i].gradY << ",";
      falseFile << falseFeatureVectors[i].gradZ << ",";
      falseFile << falseFeatureVectors[i].gradientMagnitude << ",";
      falseFile << *itg << ","; ++itg;
      falseFile << *itg << ","; ++itg;
      falseFile << *itg << std::endl;
    }
  falseFile.close();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType spacing, cip::CTType::SizeType size, 
						 cip::CTType::PointType origin, vtkSmartPointer< vtkPolyData > particles )
{
  MaskType::SizeType maskSize;
    maskSize[0] = size[0];
    maskSize[1] = size[1];
    maskSize[2] = size[2];

  MaskType::SpacingType maskSpacing;
    maskSpacing[0] = spacing[0];
    maskSpacing[1] = spacing[1];
    maskSpacing[2] = spacing[2];

  MaskType::PointType maskOrigin;
    maskOrigin[0] = origin[0];
    maskOrigin[1] = origin[1];
    maskOrigin[2] = origin[2];

  MaskType::Pointer mask = MaskType::New();
    mask->SetRegions( maskSize );
    mask->Allocate();
    mask->FillBuffer( 0 );
    mask->SetSpacing( maskSpacing );
    mask->SetOrigin( maskOrigin );

  MaskType::IndexType index;

  for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i++ )
    {
      MaskType::PointType point;
        point[0] = particles->GetPoint(i)[0];
	point[1] = particles->GetPoint(i)[1];
	point[2] = particles->GetPoint(i)[2];

      mask->TransformPhysicalPointToIndex( point, index );
      if ( mask->GetBufferedRegion().IsInside( index ) )
	{
	  mask->SetPixel( index, 1 );
	}
    }

  DistanceMapType::Pointer distanceMap = DistanceMapType::New();
    distanceMap->SetInput( mask );
    distanceMap->SetSquaredDistance( false );
    distanceMap->SetUseImageSpacing( true );
    distanceMap->SetInsideIsPositive( true );
  try
    {
    distanceMap->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught generating distance map:";
    std::cerr << excp << std::endl;
    }

  // DistanceWriterType::Pointer writer = DistanceWriterType::New();
  // writer->SetInput( distanceMap->GetOutput() );
  // writer->UseCompressionOn();
  // writer->SetFileName( "/Users/jross/tmp/foo_dist.nhdr" );
  // writer->Update();

  return distanceMap->GetOutput();
}

FEATUREVECTOR ComputeFissureFeatureVector( vtkSmartPointer< vtkPolyData > pointsParticles, cip::CTType::Pointer ct, 
					   DistanceImageType::Pointer distanceMap, const cipThinPlateSplineSurface& rhTPS,  
					   const cipThinPlateSplineSurface& roTPS,  const cipThinPlateSplineSurface& loTPS,
					   DerivativeFunctionType::Pointer derivativeFunction, 
					   HessianImageFunctionType::Pointer hessianFunction, cip::CTType::IndexType index )
{
  FEATUREVECTOR vec;

  // Mean and variance intensity values empirically found for 
  // fissures
  double meanHU = -828.0;
  double varHU  = 2091.0;

  HessianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  HessianImageFunctionType::TensorType::EigenVectorsMatrixType eigenVectors;
  HessianImageFunctionType::TensorType hessian;

  cip::CTType::PointType imPoint;
  cip::PointType point(3);

  ct->TransformIndexToPhysicalPoint( index, imPoint );

  point[0] = imPoint[0];
  point[1] = imPoint[1];
  point[2] = imPoint[2];

  vec.intensity = ct->GetPixel( index );
  vec.distanceToVessel = std::abs( distanceMap->GetPixel( index ) );

  unsigned int order[3];
  order[0] = 1; order[1] = 0; order[2] = 0;
  derivativeFunction->SetOrder( order );
  derivativeFunction->Initialize();
  vec.gradX = derivativeFunction->EvaluateAtIndex( index );

  order[0] = 0; order[1] = 1; order[2] = 0;
  derivativeFunction->SetOrder( order );
  derivativeFunction->Initialize();
  vec.gradY = derivativeFunction->EvaluateAtIndex( index );

  order[0] = 0; order[1] = 0; order[2] = 1;
  derivativeFunction->SetOrder( order );
  derivativeFunction->Initialize();
  vec.gradZ = derivativeFunction->EvaluateAtIndex( index );

  vec.gradient.push_back( vec.gradX );
  vec.gradient.push_back( vec.gradY );
  vec.gradient.push_back( vec.gradZ );
  vec.gradient.sort();

  vec.gradientMagnitude = std::sqrt(std::pow(vec.gradX, 2) + std::pow(vec.gradY, 2) + 
				    std::pow(vec.gradZ, 2));

  hessian = hessianFunction->EvaluateAtIndex( index );
  hessian.ComputeEigenAnalysis( eigenValues, eigenVectors);      
  
  vec.eigenValues.push_back( eigenValues[0] );
  vec.eigenValues.push_back( eigenValues[1] );
  vec.eigenValues.push_back( eigenValues[2] );
  vec.eigenValues.sort();
  
  for ( unsigned int i=0; i<3; i++ )
    {
      if ( eigenValues[i] == *vec.eigenValues.begin() )
	{
	  vec.eigenVector[0] = eigenVectors(i, 0);
	  vec.eigenVector[1] = eigenVectors(i, 1);
	  vec.eigenVector[2] = eigenVectors(i, 2);
	}
    }
  
  vec.eigenValueMags.push_back( std::abs(eigenValues[0]) );
  vec.eigenValueMags.push_back( std::abs(eigenValues[1]) );
  vec.eigenValueMags.push_back( std::abs(eigenValues[2]) );
  vec.eigenValueMags.sort();
  
  if ( loTPS.GetNumberSurfacePoints() > 0 )
    {
      vec.distanceToLobeSurface = cip::GetDistanceToThinPlateSplineSurface( loTPS, point );
      
      cip::VectorType normal(3);
      cip::PointType tpsPoint(3);
      
      cip::GetClosestPointOnThinPlateSplineSurface( loTPS, point, tpsPoint );
      loTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );
      
      cip::VectorType tmpVec(3);
      tmpVec[0] = vec.eigenVector[0];
      tmpVec[1] = vec.eigenVector[1];
      tmpVec[2] = vec.eigenVector[2];
      vec.angleWithLobeSurfaceNormal = cip::GetAngleBetweenVectors(normal, tmpVec, true);
    }
  else if ( roTPS.GetNumberSurfacePoints() > 0 && rhTPS.GetNumberSurfacePoints() > 0 )
    {
      double roDist = cip::GetDistanceToThinPlateSplineSurface( roTPS, point );
      double rhDist = cip::GetDistanceToThinPlateSplineSurface( rhTPS, point );
      
      double roHeight = roTPS.GetSurfaceHeight( point[0], point[1] );
      double rhHeight = rhTPS.GetSurfaceHeight( point[0], point[1] );
      
      if ( rhDist < roDist && rhHeight > roHeight )
	{
	  vec.distanceToLobeSurface = rhDist;
	  
	  cip::VectorType normal(3);
	  cip::PointType tpsPoint(3);
	  
	  cip::GetClosestPointOnThinPlateSplineSurface( rhTPS, point, tpsPoint );
	  rhTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );
	  
	  cip::VectorType tmpVec(3);
	  tmpVec[0] = vec.eigenVector[0];
	  tmpVec[1] = vec.eigenVector[1];
	  tmpVec[2] = vec.eigenVector[2];
	  
	  vec.angleWithLobeSurfaceNormal = cip::GetAngleBetweenVectors(normal, tmpVec, true);
	}
      else
	{
	  vec.distanceToLobeSurface = roDist;
	  
	  cip::VectorType normal(3);
	  cip::PointType tpsPoint(3);
	  
	  cip::GetClosestPointOnThinPlateSplineSurface( roTPS, point, tpsPoint );
	  roTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );
	  
	  cip::VectorType tmpVec(3);
	  tmpVec[0] = vec.eigenVector[0];
	  tmpVec[1] = vec.eigenVector[1];
	  tmpVec[2] = vec.eigenVector[2];
	  
	  vec.angleWithLobeSurfaceNormal = cip::GetAngleBetweenVectors(normal, tmpVec, true);
	}
    }
  else
    {
      std::cerr << "Insufficient lobe boundary surface points." << std::endl;
    }
  
  // Compute the pMeasaure, as given by equations 2 and 3 in 'Supervised Enhancement Filters : 
  // Application to Fissure Detection in Chest CT Scans' (van Rikxoort):
  if ( *vec.eigenValues.begin() < 0 )
    {
      vec.pMeasure = (*vec.eigenValueMags.rbegin() - *vec.eigenValueMags.begin())/
	(*vec.eigenValueMags.rbegin() + *vec.eigenValueMags.begin());
    }
  else
    {
      vec.pMeasure = 0;
    }
  
  // Compute the fMeasaure, as given by equation 4 in 'Supervised Enhancement Filters : 
  // Application to Fissure Detection in Chest CT Scans' (van Rikxoort):
  vec.fMeasure = std::exp( -std::pow( vec.intensity - meanHU, 2 )/(2*varHU) )*vec.pMeasure;

  return vec;
}

#endif
