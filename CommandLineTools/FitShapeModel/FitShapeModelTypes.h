#ifndef _FitShapeModelTypes_h_
#define _FitShapeModelTypes_h_

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkMatrix.h"
#include "itkPoint.h"
#include "itkImageDuplicator.h"
#include "itkTriangleMeshToBinaryImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

const unsigned int Dimension = 3; // fixed dimension
const unsigned int MaxCellDimension = 2;

// pixel types
typedef short                                                 ShortPixelType;
typedef float                                                 FloatPixelType;
typedef double                                                DoublePixelType;
typedef itk::CovariantVector< DoublePixelType, Dimension >    CovPixelType;

typedef itk::Image< ShortPixelType, Dimension >               ShortImageType;
typedef itk::Image< FloatPixelType, Dimension >               FloatImageType;
typedef itk::Image< DoublePixelType, Dimension >              DoubleImageType;

typedef itk::DefaultStaticMeshTraits< DoublePixelType, 
                                      Dimension,
                                      MaxCellDimension, 
                                      DoublePixelType, 
                                      DoublePixelType  >      MeshTraits;

typedef itk::Mesh< DoublePixelType, Dimension, MeshTraits >   MeshType;
typedef itk::MeshFileReader< MeshType >                       MeshReaderType;
typedef itk::MeshFileWriter< MeshType >                       MeshWriterType;

typedef MeshType::PointsContainer                             PointsContainer;
typedef MeshType::PointsContainerPointer                      PointsContainerPointer;
typedef MeshType::PointsContainerIterator                     PointsIterator;

typedef itk::AffineTransform< double, Dimension >             TransformType;
typedef itk::TransformFileReader                              TransformFileReaderType;
typedef itk::TransformFileWriterTemplate< double >            TransformFileWriterTemplateType;

// related basic types
typedef FloatImageType::PixelType                             PixelType;
typedef FloatImageType::IndexType                             IndexType;
typedef itk::PointSet< PixelType, Dimension >                 PointSetType;
typedef PointSetType::PointType                               PointType;

template< typename ImagePixelType >
class FitShapeModelType
{
public:
  // data, i/o, and iterator types
  typedef itk::Image< ImagePixelType, Dimension >             ImageType;
  typedef itk::Image< CovPixelType, Dimension >               CovImageType;
  typedef itk::Image< unsigned short, Dimension >             LabelMapType;
  typedef itk::ImageFileReader< ImageType >                   ImageReaderType;
  typedef itk::ImageFileWriter< ImageType >                   ImageWriterType;
  typedef itk::ImageFileWriter< LabelMapType >                LabelMapWriterType;
  typedef itk::ImageRegionConstIterator< ImageType >          ConstIteratorType;
  typedef itk::ImageRegionIterator< ImageType >               IteratorType;

  // filter types
  typedef itk::ImageDuplicator< ImageType > ImageDuplicatorType;
  typedef itk::TriangleMeshToBinaryImageFilter< MeshType, ImageType >
    TriangleMeshToBinaryImageFilterType;
  typedef itk::GradientMagnitudeRecursiveGaussianImageFilter< ImageType, ImageType >
    GradientMagnitudeRecursiveGaussianImageFilterType;
  // The output of GradientRecursiveGaussianImageFilter
  // are images of the gradient along X, Y, and Z so the type of
  // the output is a covariant vector of dimension 3 (X, Y, Z)
  typedef itk::GradientRecursiveGaussianImageFilter< ImageType, CovImageType >
    GradientRecursiveGaussianImageFilterType;
  typedef itk::VectorLinearInterpolateImageFunction< CovImageType, typename PointType::CoordRepType >
    GradientInterpolatorType;
  typedef itk::LinearInterpolateImageFunction< ImageType, typename PointType::CoordRepType >
    ImageInterpolatorType;
};

#endif
