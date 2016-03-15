#ifndef _FitShapeModelTypes_h_
#define _FitShapeModelTypes_h_

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkMatrix.h"
#include "itkPoint.h"
#include "itkTriangleMeshToBinaryImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"

const unsigned int Dimension = 3; // fixed dimension

// pixel types
typedef short                                              ShortPixelType;
typedef double                                             DoublePixelType;
typedef itk::CovariantVector< DoublePixelType, Dimension > CovPixelType;

// data, i/o, and iterator types
typedef itk::Mesh< DoublePixelType, Dimension >            MeshType;
typedef itk::Image< ShortPixelType, Dimension >            ImageType;
typedef itk::Image< DoublePixelType, Dimension >           DoubleImageType;
typedef itk::Image< CovPixelType, Dimension >              CovImageType;
typedef itk::Image< unsigned short, Dimension >            LabelMapType;
typedef itk::ImageFileReader< ImageType >                  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >                  ImageWriterType;
typedef itk::ImageFileWriter< LabelMapType >               LabelMapWriterType;
typedef itk::MeshFileReader< MeshType >                    MeshReaderType;
typedef itk::MeshFileWriter< MeshType >                    MeshWriterType;
typedef itk::ImageRegionConstIterator< ImageType >         ConstIteratorType;
typedef itk::ImageRegionIterator< ImageType >              IteratorType;

// related basic types
typedef ImageType::PixelType                       PixelType;
typedef ImageType::IndexType                       IndexType;
typedef itk::PointSet< PixelType, Dimension >      PointSetType;
typedef PointSetType::PointType                    PointType;
typedef MeshType::PointsContainer                  PointsContainer;
typedef MeshType::PointsContainerPointer           PointsContainerPointer;
typedef MeshType::PointsContainerIterator          PointsIterator;

// filter types
typedef itk::TriangleMeshToBinaryImageFilter< MeshType, ImageType >
  TriangleMeshToBinaryImageFilterType;
typedef itk::GradientMagnitudeRecursiveGaussianImageFilter< ImageType, ImageType >
  GradientMagnitudeRecursiveGaussianImageFilterType;
// The output of GradientRecursiveGaussianImageFilter
// are images of the gradient along X, Y, and Z so the type of
// the output is a covariant vector of dimension 3 (X, Y, Z)
typedef itk::GradientRecursiveGaussianImageFilter< ImageType, CovImageType >
  GradientRecursiveGaussianImageFilterType;
typedef itk::VectorLinearInterpolateImageFunction< CovImageType, PointType::CoordRepType >
  GradientInterpolatorType;

#endif
