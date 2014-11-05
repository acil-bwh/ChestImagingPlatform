#ifndef DOXYGEN_SHOULD_SKIP_THIS

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
#include "itkCastImageFilter.h"
#include "EnhanceFissuresInImageCLP.h"

typedef itk::Image< unsigned char, 3 >                                            MaskType;
typedef itk::Image< float, 3 >                                                    FloatImageType;
typedef itk::Image< float, 3 >                                                    DistanceImageType;
typedef itk::SignedMaurerDistanceMapImageFilter< MaskType, DistanceImageType >    DistanceMapType;
typedef itk::ImageRegionIteratorWithIndex< cip::CTType >                          CTIteratorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                    LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< DistanceImageType >                    DistanceImageIteratorType;
//typedef itk::DiscreteHessianGaussianImageFunction< FloatImageType >               HessianImageFunctionType;
typedef itk::DiscreteHessianGaussianImageFunction< cip::CTType >                  HessianImageFunctionType ;
typedef itk::CastImageFilter< cip::CTType, FloatImageType >                       CastType;

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType, cip::CTType::SizeType, 
						 cip::CTType::PointType, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // cip::CTType::SizeType size;
  // size[0] = 11;
  // size[1] = 11;
  // size[2] = 11;

  // cip::CTType::SpacingType spacing;
  // spacing[0] = 1;
  // spacing[1] = 1;
  // spacing[2] = 1;

  // cip::CTType::Pointer fissureImage = cip::CTType::New();
  // fissureImage->SetRegions( size );
  // fissureImage->Allocate();
  // fissureImage->SetSpacing( spacing );
  // fissureImage->FillBuffer( -1000 );

  // cip::CTType::IndexType tmpIndex;
  // tmpIndex[2] = 5;

  // for ( unsigned i=0; i<11; i++ )
  //   {
  //     tmpIndex[0] = i;
  //     for ( unsigned j=0; j<11; j++ )
  // 	{
  // 	  tmpIndex[1] = j;
  // 	  fissureImage->SetPixel( tmpIndex, -650 );
  // 	}
  //   }

  // cip::CTWriterType::Pointer foo = cip::CTWriterType::New();
  // foo->SetInput( fissureImage );
  // foo->SetFileName( "/Users/jross/Downloads/ChestImagingPlatformPrivate/Testing/Data/Input/simple_fissure.nrrd" );
  // foo->UseCompressionOn();
  // foo->Update();

  // Instatiate ChestConventions for general convenience later
  cip::ChestConventions conventions;

  // vtkPolyDataReader* fooReader = vtkPolyDataReader::New();
  //   fooReader->SetFileName( "/Users/jross/Projects/Data/Processed/COPDGene/10017X/10017X_INSP_STD_BWH_COPD/10017X_INSP_STD_BWH_COPD_rightObliqueGroundTruthParticles.vtk" );
  //   fooReader->Update();    

    //fooReader->GetOutput()->Print( std::cout );

  // std::cout << "heval0:\t" << fooReader->GetOutput()->GetPointData()->GetArray( "h0" )->GetTuple( 0 )[0] << std::endl;
  // std::cout << "heval1:\t" << fooReader->GetOutput()->GetPointData()->GetArray( "h1" )->GetTuple( 0 )[0] << std::endl;
  // std::cout << "heval2:\t" << fooReader->GetOutput()->GetPointData()->GetArray( "h2" )->GetTuple( 0 )[0] << std::endl;

  // std::cout << "Reading vessel particles..." << std::endl;
  // vtkPolyDataReader* vesselParticlesReader = vtkPolyDataReader::New();
  //   vesselParticlesReader->SetFileName( vesselParticlesFileName.c_str() );
  //   vesselParticlesReader->Update();    

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

  // CastType::Pointer caster = CastType::New();
  //   caster->SetInput( ctReader->GetOutput() );
  //   caster->Update();

  // cip::CTType::PointType point;
  //   point[0] = fooReader->GetOutput()->GetPoint( 0 )[0];
  //   point[1] = fooReader->GetOutput()->GetPoint( 0 )[1];
  //   point[2] = fooReader->GetOutput()->GetPoint( 0 )[2];

  cip::CTType::IndexType index;
  index[0] = 5;
  index[1] = 5;
  index[2] = 5;

  // ctReader->GetOutput()->TransformPhysicalPointToIndex( point, index );
  // std::cout << "Index:\t" << index << std::endl;

  HessianImageFunctionType::TensorType::EigenValuesArrayType eigenValues;
  HessianImageFunctionType::TensorType::EigenVectorsMatrixType eigenVectors;

  HessianImageFunctionType::TensorType hessian;

  unsigned int maxKernelWidth = 100;
  double variance = 1.0;
  double maxError = 0.01;

  HessianImageFunctionType::Pointer hessianFunction = HessianImageFunctionType::New();
    hessianFunction->SetUseImageSpacing( true );
    hessianFunction->SetNormalizeAcrossScale( false );
    //hessianFunction->SetInputImage( caster->GetOutput() );
    hessianFunction->SetInputImage( ctReader->GetOutput() );
    hessianFunction->SetMaximumError( maxError );
    hessianFunction->SetMaximumKernelWidth( maxKernelWidth );
    hessianFunction->SetVariance( variance );
    hessianFunction->Initialize();

  hessian = hessianFunction->EvaluateAtIndex( index );
  hessian.ComputeEigenAnalysis( eigenValues, eigenVectors);

  for ( int i=0; i<3; i++ )
    {
      std::cout << "--------------------------------" << std::endl;
      std::cout << "val:\t" << eigenValues[i] << std::endl;
      std::cout << "vec:\t";
      for ( int j=0; j<3; j++ )
	{
	  std::cout << eigenVectors(i, j) << "\t";
	}      
      std::cout << std::endl;
    }

  // cip::CTType::IndexType tmp;
  // for ( int x = -1; x<=1; x++ )
  //   {
  //     for ( int y = -1; y<=1; y++ )
  // 	{
  // 	  for ( int z = -1; z<=1; z++ )
  // 	    {
  // 	      tmp[0] = index[0] + x;
  // 	      tmp[1] = index[1] + y;
  // 	      tmp[2] = index[2] + z;
  // 	      hessian = hessianFunction->EvaluateAtIndex( tmp );
  // 	      hessian.ComputeEigenAnalysis( eigenValues, eigenVectors);
  // 	      std::cout << eigenValues << std::endl;	      
  // 	    }
  // 	}
  //   }


  // CTIteratorType cIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );
  // std::cout << "Computing..." << std::endl;
  // cIt.GoToBegin();
  // while ( !cIt.IsAtEnd() )
  //   {
  //     ++cIt;
  //   }


  // std::cout << "Getting vessel distance map..." << std::endl;
  // DistanceImageType::Pointer distanceMap = 
  //   GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
  // 			  ctReader->GetOutput()->GetOrigin(), vesselParticlesReader->GetOutput() );

  // std::cout << "Reading lung label map..." << std::endl;
  // cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
  //   labelMapReader->SetFileName( labelMapFileName );
  // try
  //   {
  //   labelMapReader->Update();
  //   }
  // catch ( itk::ExceptionObject &excp )
  //   {
  //   std::cerr << "Exception caught while reading label map:";
  //   std::cerr << excp << std::endl;
      
  //   return cip::LABELMAPREADFAILURE;
  //   }
  
  // LabelMapIteratorType lIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );
  // DistanceImageIteratorType dIt( distanceMap, distanceMap->GetBufferedRegion() );

  // double meanHU   = -828.0;
  // double varHU    = 2091.0;
  // double meanDist = 9.7;
  // double varDist  = 8.3;

  // short minCT = -950;
  // short maxCT = -650;

  // std::cout << "Enhancing fissures..." << std::endl;
  // cIt.GoToBegin();
  // lIt.GoToBegin();
  // dIt.GoToBegin();
  // while ( !cIt.IsAtEnd() )
  //   {
  //     if ( lIt.Get() > 0 )
  // 	{
  // 	  if ( cIt.Get() > minCT && cIt.Get() < maxCT )
  // 	    {
  // 	      cip::CTType::PointType point;
  // 	      ctReader->GetOutput()->TransformIndexToPhysicalPoint( cIt.GetIndex(), point );
	      
  // 	      double huTerm   = std::exp( -0.5*std::pow(cIt.Get() - meanHU, 2)/varHU );
  // 	      double distTerm = std::exp( -0.5*std::pow(std::abs(dIt.Get()) - meanDist, 2)/varDist );	      
  // 	      //double newValue = 1000.0*( huTerm*distTerm - 1.0 );
  // 	      //double newValue = -1000.0*(1.0 - huTerm*distTerm) + (cIt.Get())*huTerm*distTerm;
  // 	      double newValue = -1000.0*(1.0 - distTerm) + (cIt.Get())*distTerm;
  // 	      //double newValue = cIt.Get()*huTerm*distTerm;
  // 	      // std::cout << "-----------------------------" << std::endl;
  // 	      // std::cout << "huTerm:\t" << huTerm << std::endl;
  // 	      // std::cout << "distTerm:\t" << distTerm << std::endl;
  // 	      // std::cout << "CT:\t" << cIt.Get() << std::endl;
  // 	      // std::cout << "newValue:\t" << newValue << std::endl;
  // 	      cIt.Set( short(newValue) );
  // 	    }
  // 	  else
  // 	    {
  // 	      cIt.Set( -1000 );
  // 	    }
  // 	}      
      
  //     ++cIt;
  //     ++lIt;
  //     ++dIt;
  //   }

  // std::cout << "Writing enhanced image..." << std::endl;
  // cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
  //   writer->SetInput( ctReader->GetOutput() );
  //   writer->UseCompressionOn();
  //   writer->SetFileName( outFileName );
  // try
  //   {
  //   writer->Update();
  //   }
  // catch ( itk::ExceptionObject &excp )
  //   {
  //   std::cerr << "Exception caught writing enhanced image:";
  //   std::cerr << excp << std::endl;
  //   }

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

  return distanceMap->GetOutput();
}

#endif
