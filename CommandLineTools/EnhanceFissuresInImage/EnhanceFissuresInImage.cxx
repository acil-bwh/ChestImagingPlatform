#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "vtkPolyData.h"
#include "vtkFloatArray.h"
#include "vtkPolyDataReader.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "cipExceptionObject.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "EnhanceFissuresInImageCLP.h"

typedef itk::Image< unsigned char, 3 >                                            MaskType;
typedef itk::Image< float, 3 >                                                    DistanceImageType;
typedef itk::SignedMaurerDistanceMapImageFilter< MaskType, DistanceImageType >    DistanceMapType;
typedef itk::ImageRegionIteratorWithIndex< cip::CTType >                          CTIteratorType;
typedef itk::ImageRegionIteratorWithIndex< cip::LabelMapType >                    LabelMapIteratorType;
typedef itk::ImageRegionIteratorWithIndex< DistanceImageType >                    DistanceImageIteratorType;

DistanceImageType::Pointer GetVesselDistanceMap( cip::CTType::SpacingType, cip::CTType::SizeType, 
						 cip::CTType::PointType, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Instatiate ChestConventions for general convenience later
  cip::ChestConventions conventions;

  std::cout << "Reading vessel particles..." << std::endl;
  vtkPolyDataReader* vesselParticlesReader = vtkPolyDataReader::New();
    vesselParticlesReader->SetFileName( vesselParticlesFileName.c_str() );
    vesselParticlesReader->Update();    

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

  std::cout << "Getting vessel distance map..." << std::endl;
  DistanceImageType::Pointer distanceMap = 
    GetVesselDistanceMap( ctReader->GetOutput()->GetSpacing(), ctReader->GetOutput()->GetBufferedRegion().GetSize(), 
			  ctReader->GetOutput()->GetOrigin(), vesselParticlesReader->GetOutput() );

  std::cout << "Reading lung label map..." << std::endl;
  cip::LabelMapReaderType::Pointer labelMapReader = cip::LabelMapReaderType::New();
    labelMapReader->SetFileName( labelMapFileName );
  try
    {
    labelMapReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught while reading label map:";
    std::cerr << excp << std::endl;
      
    return cip::LABELMAPREADFAILURE;
    }
  
  CTIteratorType cIt( ctReader->GetOutput(), ctReader->GetOutput()->GetBufferedRegion() );
  LabelMapIteratorType lIt( labelMapReader->GetOutput(), labelMapReader->GetOutput()->GetBufferedRegion() );
  DistanceImageIteratorType dIt( distanceMap, distanceMap->GetBufferedRegion() );

  double meanHU   = -828.0;
  double varHU    = 2091.0;
  double meanDist = 9.7;
  double varDist  = 8.3;

  short minCT = -950;
  short maxCT = -650;

  std::cout << "Enhancing fissures..." << std::endl;
  cIt.GoToBegin();
  lIt.GoToBegin();
  dIt.GoToBegin();
  while ( !cIt.IsAtEnd() )
    {
      if ( lIt.Get() > 0 )
	{
	  if ( cIt.Get() > minCT && cIt.Get() < maxCT )
	    {
	      cip::CTType::PointType point;
	      ctReader->GetOutput()->TransformIndexToPhysicalPoint( cIt.GetIndex(), point );
	      
	      double huTerm   = std::exp( -0.5*std::pow(cIt.Get() - meanHU, 2)/varHU );
	      double distTerm = std::exp( -0.5*std::pow(std::abs(dIt.Get()) - meanDist, 2)/varDist );	      
	      //double newValue = 1000.0*( huTerm*distTerm - 1.0 );
	      //double newValue = -1000.0*(1.0 - huTerm*distTerm) + (cIt.Get())*huTerm*distTerm;
	      double newValue = -1000.0*(1.0 - distTerm) + (cIt.Get())*distTerm;
	      //double newValue = cIt.Get()*huTerm*distTerm;
	      // std::cout << "-----------------------------" << std::endl;
	      // std::cout << "huTerm:\t" << huTerm << std::endl;
	      // std::cout << "distTerm:\t" << distTerm << std::endl;
	      // std::cout << "CT:\t" << cIt.Get() << std::endl;
	      // std::cout << "newValue:\t" << newValue << std::endl;
	      cIt.Set( short(newValue) );
	    }
	  else
	    {
	      cIt.Set( -1000 );
	    }
	}      
      
      ++cIt;
      ++lIt;
      ++dIt;
    }

  std::cout << "Writing enhanced image..." << std::endl;
  cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
    writer->SetInput( ctReader->GetOutput() );
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
