#ifndef DOXYGEN_SHOULD_SKIP_THIS

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
#include "itkDiscreteHessianGaussianImageFunction.h"
#include "itkDiscreteGaussianDerivativeImageFunction.h"
#include "cipLobeSurfaceModelIO.h"
#include "EnhanceFissuresInImageCLP.h"

typedef itk::Image< unsigned char, 3 >                               MaskType;
typedef itk::Image< float, 3 >                                       FloatImageType;
typedef itk::ImageRegionIteratorWithIndex< cip::CTType >             CTIteratorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >       LabelMapIteratorType;
typedef itk::DiscreteHessianGaussianImageFunction< cip::CTType >     HessianImageFunctionType;
typedef itk::DiscreteGaussianDerivativeImageFunction< cip::CTType >  DerivativeFunctionType;

struct FEATUREVECTORINFO
{
  double eigenVector[3];
  short  intensity;
  double distanceToLobeSurface;
  double angleWithLobeSurfaceNormal;
  double pMeasure;
  double fMeasure;
  std::list< double > eigenValues;
  std::list< double > eigenValueMags;
  bool matchesLO;
  bool matchesRO;
  bool matchesRH;
  double gradX;
  double gradY;
  double gradZ;
  double gradMin;
  double gradMid;
  double gradMax;
  double gradientMagnitude;
};

void UpdateFeatureVectorWithShapeModelInfo( cip::CTType::PointType, FEATUREVECTORINFO&,
					    const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&, 
					    const cipThinPlateSplineSurface&, bool, bool );
void UpdateFeatureVectorWithHessianInfo( cip::CTType::IndexType, cip::CTType::PointType, 
					 HessianImageFunctionType::Pointer,
					 const cipThinPlateSplineSurface&, const cipThinPlateSplineSurface&,  
					 const cipThinPlateSplineSurface&, FEATUREVECTORINFO& );
void UpdateFeatureVectorWithGradientInfo( cip::CTType::IndexType, DerivativeFunctionType::Pointer, 
					  FEATUREVECTORINFO& );
double GetIntensityAndShapeModelFeaturesProbability( const FEATUREVECTORINFO& );
double GetIntensityShapeModelAndHessianFeaturesProbability( FEATUREVECTORINFO& );
double GetIntensityShapeModelHessianAndGradientFeaturesProbability( FEATUREVECTORINFO& );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cip::ChestConventions conventions;

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
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
  try
    {
    labelMapReader->Update();
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

  // Allocation space for the output image
  cip::CTType::Pointer outImage = cip::CTType::New();
    outImage->SetRegions( ctReader->GetOutput()->GetBufferedRegion().GetSize() );
    outImage->Allocate();
    outImage->FillBuffer( -1000 );
    outImage->SetSpacing( ctReader->GetOutput()->GetSpacing() );
    outImage->SetOrigin( ctReader->GetOutput()->GetOrigin() );

  CTIteratorType ctIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );
  CTIteratorType outIt( outImage, outImage->GetBufferedRegion() );
  LabelMapIteratorType lIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );

  std::list< double >::iterator eigenValIt;
  
  std::cout << "Enhancing fissures..." << std::endl;
  ctIt.GoToBegin();
  lIt.GoToBegin();
  outIt.GoToBegin();

  double prob;
  unsigned char cipRegion;
  cip::CTType::SizeType size = ctReader->GetOutput()->GetBufferedRegion().GetSize();
  cip::CTType::PointType imPoint;
  bool isLeftLung;
  bool isRightLung;

  int lastZ = -1;
  while ( !ctIt.IsAtEnd() )
    {
      if ( lIt.Get() != 0 && ctIt.Get() < -650 )
	{
	  cipRegion = conventions.GetChestRegionFromValue( lIt.Get() );
	  isLeftLung  = conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::LEFTLUNG) );
	  isRightLung = conventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, (unsigned char)(cip::RIGHTLUNG) );
	  if ( (isLeftLung && loTPS.GetNumberSurfacePoints() > 0) ||
	       (isRightLung && roTPS.GetNumberSurfacePoints() > 0 && rhTPS.GetNumberSurfacePoints() > 0) )
	    {
	      ctReader->GetOutput()->TransformIndexToPhysicalPoint( ctIt.GetIndex(), imPoint );
	      
	      FEATUREVECTORINFO vec;
	      vec.intensity = ctIt.Get();
	      
	      UpdateFeatureVectorWithShapeModelInfo( imPoint, vec, rhTPS, roTPS, loTPS, isLeftLung, isRightLung );
	      prob = GetIntensityAndShapeModelFeaturesProbability( vec );
	      
	      // The following probability threshold corresponds to a 0.995 TPR. Only consider
	      // points with at least this probability of being a fissure helps reduce computation
	      // (we only compute additional features for those points that are likely candidates)
	      if ( prob > 0.111636111749 )
		{
		  UpdateFeatureVectorWithHessianInfo( ctIt.GetIndex(), imPoint, hessianFunction,
						      rhTPS, roTPS, loTPS, vec );
		  if ( *vec.eigenValues.begin() < 0 )
		    {
		      prob = GetIntensityShapeModelAndHessianFeaturesProbability( vec );
		      if ( prob > 0.0441242935955 )
			{
			  UpdateFeatureVectorWithGradientInfo( ctIt.GetIndex(), derivativeFunction, vec );
			  prob = GetIntensityShapeModelHessianAndGradientFeaturesProbability( vec );
			  
			  short newVal = short(prob*(double(ctIt.Get()) + 1000.0) - 1000.0);
			  outIt.Set( newVal );
			}
		    }
		}
	    }
	}
    
      // if ( ctIt.GetIndex()[2] != lastZ )
      // 	{
      // 	  lastZ = ctIt.GetIndex()[2];
      // 	  std::cout << lastZ << std::endl;
      // 	}

      ++outIt;
      ++ctIt;
      ++lIt;
    }

  if ( outFileName.compare("NA") != 0 )
    {
      std::cout << "Writing enhanced image..." << std::endl;
      cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
        writer->SetInput( outImage );
  	writer->UseCompressionOn();
  	writer->SetFileName( outFileName );
      try
  	{
  	  writer->Update();
  	}
      catch ( itk::ExceptionObject &excp )
  	{
  	  std::cerr << "Exception caught writing enhanced image:";
  	  std::cerr << excp << std::endl;
  	}
    }    

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void UpdateFeatureVectorWithShapeModelInfo( cip::CTType::PointType imPoint,
					    FEATUREVECTORINFO& vec,
					    const cipThinPlateSplineSurface& rhTPS,  
					    const cipThinPlateSplineSurface& roTPS,  
					    const cipThinPlateSplineSurface& loTPS,
					    bool isLeftLung, bool isRightLung )
{
  cip::PointType point(3);
    point[0] = imPoint[0];
    point[1] = imPoint[1];
    point[2] = imPoint[2];
  
  if ( loTPS.GetNumberSurfacePoints() > 0 && isLeftLung )
    {
      vec.distanceToLobeSurface = cip::GetDistanceToThinPlateSplineSurface( loTPS, point );
      vec.matchesLO = true;
      vec.matchesRO = false;
      vec.matchesRH = false;
    }
  else if ( roTPS.GetNumberSurfacePoints() > 0 && rhTPS.GetNumberSurfacePoints() > 0 && isRightLung )
    {
      double roDist = cip::GetDistanceToThinPlateSplineSurface( roTPS, point );
      double rhDist = cip::GetDistanceToThinPlateSplineSurface( rhTPS, point );
      
      double roHeight = roTPS.GetSurfaceHeight( point[0], point[1] );
      double rhHeight = rhTPS.GetSurfaceHeight( point[0], point[1] );
      
      if ( rhDist < roDist && rhHeight > roHeight )
  	{
  	  vec.distanceToLobeSurface = rhDist;
	  vec.matchesLO = false;
	  vec.matchesRO = false;
	  vec.matchesRH = true;
  	}
      else
  	{
  	  vec.distanceToLobeSurface = roDist;
	  vec.matchesLO = false;
	  vec.matchesRO = true;
	  vec.matchesRH = false;	  
  	}
    }
}

void UpdateFeatureVectorWithHessianInfo( cip::CTType::IndexType index, cip::CTType::PointType imPoint, 
					 HessianImageFunctionType::Pointer hessianFunction,
					 const cipThinPlateSplineSurface& rhTPS,  
					 const cipThinPlateSplineSurface& roTPS,  
					 const cipThinPlateSplineSurface& loTPS, FEATUREVECTORINFO& vec )
{
  cip::PointType point(3);
    point[0] = imPoint[0];
    point[1] = imPoint[1];
    point[2] = imPoint[2];

  HessianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  HessianImageFunctionType::TensorType::EigenVectorsMatrixType eigenVectors;
  HessianImageFunctionType::TensorType hessian;

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

  cip::VectorType normal(3);
  cip::PointType tpsPoint(3);
  if ( vec.matchesLO )
    {
      cip::GetClosestPointOnThinPlateSplineSurface( loTPS, point, tpsPoint );
      loTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );      
    }
  else if ( vec.matchesRH )
    {
      cip::GetClosestPointOnThinPlateSplineSurface( rhTPS, point, tpsPoint );
      rhTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );	  
    }
  else
    {	  
      cip::GetClosestPointOnThinPlateSplineSurface( roTPS, point, tpsPoint );
      roTPS.GetSurfaceNormal( tpsPoint[0], tpsPoint[1], normal );	  
    }

  cip::VectorType tmpVec(3);
    tmpVec[0] = vec.eigenVector[0];
    tmpVec[1] = vec.eigenVector[1];
    tmpVec[2] = vec.eigenVector[2];

  vec.angleWithLobeSurfaceNormal = cip::GetAngleBetweenVectors(normal, tmpVec, true);
  
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
  double meanHU = -828.0;
  double varHU  = 2091.0;
  vec.fMeasure = std::exp( -std::pow( vec.intensity - meanHU, 2 )/(2*varHU) )*vec.pMeasure;
}

void UpdateFeatureVectorWithGradientInfo( cip::CTType::IndexType index, DerivativeFunctionType::Pointer derivativeFunction, 
					  FEATUREVECTORINFO& vec )
{
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

  std::list< double > tmpList;
    tmpList.push_back( vec.gradX );
    tmpList.push_back( vec.gradY );
    tmpList.push_back( vec.gradZ );
    tmpList.sort();

  std::list< double >::iterator lit = tmpList.begin();
  vec.gradMin = *lit; ++lit;
  vec.gradMid = *lit; ++lit;
  vec.gradMax = *lit;

  vec.gradientMagnitude = std::sqrt(std::pow(vec.gradX, 2) + std::pow(vec.gradY, 2) + 
				    std::pow(vec.gradZ, 2));
}

double GetIntensityAndShapeModelFeaturesProbability( const FEATUREVECTORINFO& vec )
{
  double intercept                = 0.93411054;
  double intensity_co             = -0.00092705723252;
  double distanceToLobeSurface_co = -0.111715453128;

  double intensity             = double(vec.intensity);
  double distanceToLobeSurface = vec.distanceToLobeSurface;

  double expArg =
    intercept +
    intensity*intensity_co +
    distanceToLobeSurface*distanceToLobeSurface_co;

  return 1.0/(1.0 + std::exp(-expArg));
}

double GetIntensityShapeModelAndHessianFeaturesProbability( FEATUREVECTORINFO& vec )
{
  double intercept                     = 1.06413733;
  double eigenValue0_co                = -0.0653800014727;
  double eigenValue1_co                = 0.0755923725566;
  double eigenValue2_co                = -0.0782643785069;
  double eigenValueMag0_co             = -0.0249833783353;
  double eigenValueMag1_co             = -0.0207810237335;
  double eigenValueMag2_co             = 0.0252331359284;
  double intensity_co                  = -6.9407730482e-05;
  double distanceToLobeSurface_co      = -0.10760811744;
  double angleWithLobeSurfaceNormal_co = -0.0697038588434;
  double pMeasure_co                   = 1.2563839066;
  double fMeasure_co                   = 3.1864152494;

  std::list< double >::iterator vIt = vec.eigenValues.begin();
  std::list< double >::iterator mIt = vec.eigenValueMags.begin();

  double intensity                  = double(vec.intensity);
  double distanceToLobeSurface      = vec.distanceToLobeSurface;
  double eigenValue0                = *vIt; ++vIt;
  double eigenValue1                = *vIt; ++vIt;
  double eigenValue2                = *vIt;
  double eigenValueMag0             = *mIt; ++mIt;
  double eigenValueMag1             = *mIt; ++mIt;
  double eigenValueMag2             = *mIt;
  double angleWithLobeSurfaceNormal = vec.angleWithLobeSurfaceNormal;
  double pMeasure                   = vec.pMeasure;
  double fMeasure                   = vec.fMeasure;

  double expArg =
    intercept +
    intensity*intensity_co +
    distanceToLobeSurface*distanceToLobeSurface_co +
    eigenValue0*eigenValue0_co +
    eigenValue1*eigenValue1_co +
    eigenValue2*eigenValue2_co +
    eigenValueMag0*eigenValueMag0_co +
    eigenValueMag1*eigenValueMag1_co +
    eigenValueMag2*eigenValueMag2_co +
    intensity*intensity_co +
    distanceToLobeSurface*distanceToLobeSurface_co +
    angleWithLobeSurfaceNormal*angleWithLobeSurfaceNormal_co +
    pMeasure*pMeasure_co +
    fMeasure*fMeasure_co; 

  return 1.0/(1.0 + std::exp(-expArg));
}

double GetIntensityShapeModelHessianAndGradientFeaturesProbability( FEATUREVECTORINFO& vec )
{
  double intercept                     =  0.28011861;
  double eigenValue0_co                = -0.0940888285617;
  double eigenValue1_co                =  0.0875079537809;
  double eigenValue2_co                = -0.0545950818784;
  double eigenValueMag0_co             = -0.0502639880289;
  double eigenValueMag1_co             = -0.0128664033151;
  double eigenValueMag2_co             =  0.0384048373985;
  double intensity_co                  = -0.000617640861664;
  double distanceToLobeSurface_co      = -0.0972966395015;
  double angleWithLobeSurfaceNormal_co = -0.0619096106263;
  double pMeasure_co                   =  0.859674927159;
  double fMeasure_co                   =  2.83452260844;
  double gradX_co                      = -0.00580886202778;
  double gradY_co                      = -0.000232390016618;
  double gradZ_co                      = -0.00301265941143;
  double gradientMagnitude_co          = -0.0282271325883;
  double gradMin_co                    =  0.0200687184842;
  double gradMid_co                    =  0.015902780787;
  double gradMax_co                    = -0.0230877084503;

  std::list< double >::iterator vIt = vec.eigenValues.begin();
  std::list< double >::iterator mIt = vec.eigenValueMags.begin();

  double intensity                  = double(vec.intensity);
  double distanceToLobeSurface      = vec.distanceToLobeSurface;
  double eigenValue0                = *vIt; ++vIt;
  double eigenValue1                = *vIt; ++vIt;
  double eigenValue2                = *vIt;
  double eigenValueMag0             = *mIt; ++mIt;
  double eigenValueMag1             = *mIt; ++mIt;
  double eigenValueMag2             = *mIt;
  double angleWithLobeSurfaceNormal = vec.angleWithLobeSurfaceNormal;
  double pMeasure                   = vec.pMeasure;
  double fMeasure                   = vec.fMeasure;
  double gradX                      = vec.gradX;
  double gradY                      = vec.gradY;
  double gradZ                      = vec.gradZ;
  double gradientMagnitude          = vec.gradientMagnitude;
  double gradMin                    = vec.gradMin;
  double gradMid                    = vec.gradMid;
  double gradMax                    = vec.gradMax;

  double expArg =
    intercept +
    intensity*intensity_co +
    distanceToLobeSurface*distanceToLobeSurface_co +
    eigenValue0*eigenValue0_co +
    eigenValue1*eigenValue1_co +
    eigenValue2*eigenValue2_co +
    eigenValueMag0*eigenValueMag0_co +
    eigenValueMag1*eigenValueMag1_co +
    eigenValueMag2*eigenValueMag2_co +
    intensity*intensity_co +
    distanceToLobeSurface*distanceToLobeSurface_co +
    angleWithLobeSurfaceNormal*angleWithLobeSurfaceNormal_co +
    pMeasure*pMeasure_co +
    fMeasure*fMeasure_co +
    gradX*gradX_co +
    gradY*gradY_co +
    gradZ*gradZ_co +
    gradientMagnitude*gradientMagnitude_co +
    gradMin*gradMin_co +
    gradMid*gradMid_co +
    gradMax*gradMax_co;

  return 1.0/(1.0 + std::exp(-expArg));
}

#endif
