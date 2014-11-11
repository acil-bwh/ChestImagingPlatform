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
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkDiscreteHessianGaussianImageFunction.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipVesselParticleConnectedComponentFilter.h"
#include "EnhanceFissuresInImageCLP.h"

typedef itk::Image< unsigned char, 3 >                                            MaskType;
typedef itk::Image< float, 3 >                                                    FloatImageType;
typedef itk::Image< float, 3 >                                                    DistanceImageType;
typedef itk::ImageFileWriter< DistanceImageType >                                 DistanceWriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< MaskType, DistanceImageType >    DistanceMapType;
typedef itk::ImageRegionIteratorWithIndex< cip::CTType >                          CTIteratorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                    LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< DistanceImageType >                    DistanceImageIteratorType;
typedef itk::DiscreteHessianGaussianImageFunction< cip::CTType >                  HessianImageFunctionType ;

struct FEATUREVECTOR
{
  double eigenVector[3];
  short  intensity;
  double distanceToVessel;
  double distanceToLobeSurface;
  double angleWithLobeSurfaceNormal;
  double pMeasure;
  double fMeasure;
  std::list< double > eigenValues;
  std::list< double > eigenValueMags;
};

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType, cip::CTType::SizeType, 
						 cip::CTType::PointType, vtkSmartPointer< vtkPolyData > );
FEATUREVECTOR ComputeFissureFeatureVector( cip::CTType::IndexType, cip::CTType::Pointer, 
					   DistanceImageType::Pointer, const cipThinPlateSplineSurface&,  
					   const cipThinPlateSplineSurface&,  const cipThinPlateSplineSurface&,
					   HessianImageFunctionType::Pointer );
double GetFissureProbability( FEATUREVECTOR );

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
    filter->SetInterParticleSpacing( interParticleSpacing );
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
  DistanceImageType::Pointer distanceMap = DistanceImageType::New();
  distanceMap = 
    GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
  			  ctReader->GetOutput()->GetOrigin(), filter->GetOutput() );

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

  unsigned int inc = 0;
  cip::CTType::SizeType size = ctReader->GetOutput()->GetBufferedRegion().GetSize();

  while ( !ctIt.IsAtEnd() )
    {
      if ( ctIt.GetIndex()[2] > 300 && ctIt.GetIndex()[2] < 400 && ctIt.GetIndex()[1] > 260 && ctIt.GetIndex()[1] < 320 && ctIt.GetIndex()[0] > 80 && ctIt.GetIndex()[0] < 120 )
	{
	  if ( lIt.Get() != 0 && ctIt.Get() < -650 )
	    {
	      FEATUREVECTOR vec = ComputeFissureFeatureVector( ctIt.GetIndex(), ctReader->GetOutput(), distanceMap, 
							       rhTPS, roTPS, loTPS, hessianFunction );
	      if ( *vec.eigenValues.begin() < 0 && vec.distanceToVessel > 2 )
		{
		  double prob = GetFissureProbability( vec );
		  outIt.Set( short(-1000.0*(1.0 - prob) + prob*(double(ctIt.Get()) + 0.0)));
		  
		  if ( inc % 50000 == 0 )
		    {
		      std::cout << double(inc)/double(size[0]*size[1]*size[2]) << std::endl;
		    }
		}	
	    }
	}

      inc++;
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
      mask->SetPixel( index, 1 );
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

  DistanceWriterType::Pointer writer = DistanceWriterType::New();
  writer->SetInput( distanceMap->GetOutput() );
  writer->UseCompressionOn();
  writer->SetFileName( "/Users/jross/tmp/foo_dist.nhdr" );
  writer->Update();

  return distanceMap->GetOutput();
}

FEATUREVECTOR ComputeFissureFeatureVector( cip::CTType::IndexType index, cip::CTType::Pointer ct, 
					   DistanceImageType::Pointer distanceMap, const cipThinPlateSplineSurface& rhTPS,  
					   const cipThinPlateSplineSurface& roTPS,  const cipThinPlateSplineSurface& loTPS,
					   HessianImageFunctionType::Pointer hessianFunction )
{
  FEATUREVECTOR vec;

  // Mean and variance intensity values empirically found for 
  // fissures
  double meanHU = -828.0;
  double varHU  = 2091.0;

  HessianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  HessianImageFunctionType::TensorType::EigenVectorsMatrixType eigenVectors;
  HessianImageFunctionType::TensorType hessian;

  cip::PointType point(3);
  cip::CTType::PointType imPoint;
  ct->TransformIndexToPhysicalPoint( index, imPoint );

  point[0] = imPoint[0];
  point[1] = imPoint[1];
  point[2] = imPoint[2];

  vec.intensity = ct->GetPixel( index );
  vec.distanceToVessel = std::abs( distanceMap->GetPixel( index ) );

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

double GetFissureProbability( FEATUREVECTOR vec )
{
  // The following values were learned from a training set
  // taken from COPDGene data
  double intercept                     =  1.20702066;
  double eigenValue0_co                = -0.0838340285296;
  double eigenValue1_co                =  0.0646291719633;
  double eigenValue2_co                = -0.0612478543215;
  double eigenValueMag0_co             = -0.0355337954651;
  double eigenValueMag1_co             = -0.0260964307701;
  double eigenValueMag2_co             =  0.0121555946598;
  double intensity_co                  = -0.000514098310704;
  double distanceToVessel_co           =  0.0497639090219;
  double distanceToLobeSurface_co      = -0.103594456986;
  double angleWithLobeSurfaceNormal_co = -0.064595541012;
  double pMeasure_co                   =  0.687978749475;
  double fMeasure_co                   =  3.23773903683;

  std::list<double>::iterator itv = vec.eigenValues.begin();
  std::list<double>::iterator itm = vec.eigenValueMags.begin();

  double eigenValue0                = *itv; ++itv;
  double eigenValue1                = *itv; ++itv;
  double eigenValue2                = *itv;
  double eigenValueMag0             = *itm; ++itm;
  double eigenValueMag1             = *itm; ++itm;
  double eigenValueMag2             = *itm;
  double intensity                  = double(vec.intensity);
  double distanceToVessel           = vec.distanceToVessel;
  double distanceToLobeSurface      = vec.distanceToLobeSurface;
  double angleWithLobeSurfaceNormal = vec.angleWithLobeSurfaceNormal;
  double pMeasure                   = vec.pMeasure;
  double fMeasure                   = vec.fMeasure;

  std::list< double > eigenValues;
  std::list< double > eigenValueMags;  

  double expArg =
    intercept +
    eigenValue0*eigenValue0_co +
    eigenValue1*eigenValue1_co +
    eigenValue2*eigenValue2_co +
    eigenValueMag0*eigenValueMag0_co +
    eigenValueMag1*eigenValueMag1_co +
    eigenValueMag2*eigenValueMag2_co +
    intensity*intensity_co +
    distanceToVessel*distanceToVessel_co +
    distanceToLobeSurface*distanceToLobeSurface_co +
    angleWithLobeSurfaceNormal*angleWithLobeSurfaceNormal_co +
    pMeasure*pMeasure_co +
    fMeasure*fMeasure_co;

  return 1.0/(1.0 + std::exp(-expArg));
}

#endif
