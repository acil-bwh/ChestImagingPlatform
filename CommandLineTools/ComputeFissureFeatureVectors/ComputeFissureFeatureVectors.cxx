#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <fstream>
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

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType, cip::CTType::SizeType, 
						 cip::CTType::PointType, vtkSmartPointer< vtkPolyData > );

struct FEATUREVECTOR
{
  double smallestEigenValue;
  double middleEigenValue;
  double largestEigenValue;
  double eigenVector[3];
  short  intensity;
  double distanceToVessel;
  double distanceToLobeSurface;
  double angleWithLobeSurfaceNormal;
};

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
      rightShapeModelIO.Read();

      rhTPS.SetSurfacePoints( rightShapeModelIO.GetOutput()->GetMeanRightHorizontalSurfacePoints() );
      roTPS.SetSurfacePoints( rightShapeModelIO.GetOutput()->GetMeanRightObliqueSurfacePoints() );
    }
  else if ( leftShapeModelFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left shape model..." << std::endl;
      cip::LobeSurfaceModelIO leftShapeModelIO;
      leftShapeModelIO.SetFileName( leftShapeModelFileName );
      leftShapeModelIO.Read();

      loTPS.SetSurfacePoints( leftShapeModelIO.GetOutput()->GetMeanSurfacePoints() );
    }
  else 
    {
      std::cerr << "Must specify a shape model file name." << std::endl;
      return 1;
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

  std::cout << "Getting vessel distance map..." << std::endl;
  DistanceImageType::Pointer distanceMap = DistanceImageType::New();
  distanceMap = 
    GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
  			  ctReader->GetOutput()->GetOrigin(), vesselParticles );

  std::cout << "Reading particles file indicating where to compute feature vectors..." << std::endl;
  vtkSmartPointer< vtkPolyDataReader > pointsParticlesReader = vtkPolyDataReader::New();
    pointsParticlesReader->SetFileName( pointsParticlesFileName.c_str() );
    pointsParticlesReader->Update();    

  HessianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  HessianImageFunctionType::TensorType::EigenVectorsMatrixType eigenVectors;

  HessianImageFunctionType::TensorType hessian;

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

  // Now loop through the query points and compute the feature vectors
  cip::CTType::PointType imPoint;
  cip::CTType::IndexType index;
  cip::PointType point(3);

  std::vector< FEATUREVECTOR > featureVectors;
  std::cout << "Computing feature vectors..." << std::endl;
  for ( unsigned int i=0; i<pointsParticlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
      FEATUREVECTOR vec;

      imPoint[0] = pointsParticlesReader->GetOutput()->GetPoint(i)[0];
      imPoint[1] = pointsParticlesReader->GetOutput()->GetPoint(i)[1];
      imPoint[2] = pointsParticlesReader->GetOutput()->GetPoint(i)[2];

      point[0] = imPoint[0];
      point[1] = imPoint[1];
      point[2] = imPoint[2];

      ctReader->GetOutput()->TransformPhysicalPointToIndex( imPoint, index );
      vec.intensity = ctReader->GetOutput()->GetPixel( index );
      vec.distanceToVessel = abs( distanceMap->GetPixel( index ) );

      hessian = hessianFunction->EvaluateAtIndex( index );
      hessian.ComputeEigenAnalysis( eigenValues, eigenVectors);      
      
      if ( eigenValues[0] < eigenValues[1] < eigenValues[2] )
  	{
  	  vec.smallestEigenValue = eigenValues[0];
  	  vec.middleEigenValue   = eigenValues[1];
  	  vec.largestEigenValue = eigenValues[2];
  	  vec.eigenVector[0] = eigenVectors(0, 0);
  	  vec.eigenVector[1] = eigenVectors(0, 1);
  	  vec.eigenVector[2] = eigenVectors(0, 2);
  	}
      else if ( eigenValues[0] < eigenValues[2] < eigenValues[1] )
  	{
  	  vec.smallestEigenValue = eigenValues[0];
  	  vec.middleEigenValue   = eigenValues[2];
  	  vec.largestEigenValue = eigenValues[1];
  	  vec.eigenVector[0] = eigenVectors(0, 0);
  	  vec.eigenVector[1] = eigenVectors(0, 1);
  	  vec.eigenVector[2] = eigenVectors(0, 2);
  	}
      else if ( eigenValues[1] < eigenValues[0] < eigenValues[2] )
  	{
  	  vec.smallestEigenValue = eigenValues[1];
  	  vec.middleEigenValue   = eigenValues[0];
  	  vec.largestEigenValue = eigenValues[2];
  	  vec.eigenVector[0] = eigenVectors(1, 0);
  	  vec.eigenVector[1] = eigenVectors(1, 1);
  	  vec.eigenVector[2] = eigenVectors(1, 2);
  	}
      else if ( eigenValues[1] < eigenValues[2] < eigenValues[0] )
  	{
  	  vec.smallestEigenValue = eigenValues[1];
  	  vec.middleEigenValue   = eigenValues[2];
  	  vec.largestEigenValue = eigenValues[0];
  	  vec.eigenVector[0] = eigenVectors(1, 0);
  	  vec.eigenVector[1] = eigenVectors(1, 1);
  	  vec.eigenVector[2] = eigenVectors(1, 2);
  	}
      else if ( eigenValues[2] < eigenValues[0] < eigenValues[1] )
  	{
  	  vec.smallestEigenValue = eigenValues[2];
  	  vec.middleEigenValue   = eigenValues[0];
  	  vec.largestEigenValue = eigenValues[1];
  	  vec.eigenVector[0] = eigenVectors(2, 0);
  	  vec.eigenVector[1] = eigenVectors(2, 1);
  	  vec.eigenVector[2] = eigenVectors(2, 2);
  	}
      else if ( eigenValues[2] < eigenValues[1] < eigenValues[0] )
  	{
  	  vec.smallestEigenValue = eigenValues[2];
  	  vec.middleEigenValue   = eigenValues[1];
  	  vec.largestEigenValue = eigenValues[0];
  	  vec.eigenVector[0] = eigenVectors(2, 0);
  	  vec.eigenVector[1] = eigenVectors(2, 1);
  	  vec.eigenVector[2] = eigenVectors(2, 2);
  	}

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
  	  return 1;
  	}

      featureVectors.push_back( vec );
    }

  std::cout << "Writing feature vectors to file..." << std::endl;
  std::ofstream file( outFileName.c_str() );

  file << "smallestEigenValue,middleEigenValue,largestEigenValue,intensity,";
  file << "distanceToVessel,distanceToLobeSurface,angleWithLobeSurfaceNormal" << std::endl;
  for ( unsigned int i=0; i<featureVectors.size(); i++ )
    {
      file << featureVectors[i].smallestEigenValue << ",";
      file << featureVectors[i].middleEigenValue << ",";
      file << featureVectors[i].largestEigenValue << ",";
      file << featureVectors[i].intensity << ",";
      file << featureVectors[i].distanceToVessel << ",";
      file << featureVectors[i].distanceToLobeSurface << ",";
      file << featureVectors[i].angleWithLobeSurfaceNormal << std::endl;
    }
  file.close();

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

#endif
