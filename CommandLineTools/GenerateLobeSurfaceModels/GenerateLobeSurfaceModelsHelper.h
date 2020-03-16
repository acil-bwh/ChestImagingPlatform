#ifndef __GenerateLobeSurfaceModelsHelper_h
#define __GenerateLobeSurfaceModelsHelper_h

#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageRegistrationMethod.h"
#include "itkKappaStatisticImageToImageMetric.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "cipChestConventions.h"
#include "cipThinPlateSplineSurface.h"
#include "itkNrrdImageIO.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkIdentityTransform.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"
#include "itkContinuousIndex.h"
#include "cipChestRegionChestTypeLocationsIO.h"
#include "cipLobeSurfaceModel.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipHelper.h"

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
typedef itk::CIPExtractChestLabelMapImageFilter< 3 >                                LabelMapExtractorType;
typedef const itk::TransformFileReader::TransformListType*                                TransformListType;
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

void ReadFissurePointsFromFile( std::string, std::vector< ImageType::PointType >*,
				std::vector< ImageType::PointType >*, std::vector< ImageType::PointType >* );

void GetDomainPoints( std::vector< std::vector< ImageType::PointType > >,
		      std::vector< ImageType::PointType >*, ImageType::Pointer, int );

void GetZValuesFromTPS( std::vector< ImageType::PointType >, std::vector< double >*,
			std::vector< ImageType::PointType >, ImageType::Pointer );

PCA GetHighDimensionPCA( std::vector< std::vector< double > > );

PCA GetLowDimensionPCA( std::vector< std::vector< double > > );

void GetDomainPatternedPoints(ImageType::PointType, float[2], unsigned int, 
			      std::vector< ImageType::PointType >* );

#endif
