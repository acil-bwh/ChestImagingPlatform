/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFrangiTubularnessFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFrangiTubularnessFeatureGenerator_h
#define __itkFrangiTubularnessFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkFrangiTubularnessImageFilter.h"

namespace itk
{

/** \class FrangiTubularnessFeatureGenerator
 * \brief Generates a feature image by computing measures based on the Hessian Eigenvalues.
 *
 * The typical use of this class would be to generate a map of {blobs, tubes, sheets}.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT FrangiTubularnessFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef FrangiTubularnessFeatureGenerator          Self;
  typedef FeatureGenerator<NDimension>               Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FrangiTubularnessFeatureGenerator, FeatureGenerator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input to this
   * feature generator. */
  typedef signed short                                      InputPixelType;
  typedef Image< InputPixelType, Dimension >                InputImageType;
  typedef ImageSpatialObject< NDimension, InputPixelType >  InputImageSpatialObjectType;
  typedef typename InputImageSpatialObjectType::Pointer     InputImageSpatialObjectPointer;
  typedef typename Superclass::SpatialObjectType            SpatialObjectType;

  /** Input data that will be used for generating the feature. */
  using ProcessObject::SetInput;
  void SetInput( const SpatialObjectType * input );
  const SpatialObjectType * GetInput() const;

  /** Output data that carries the feature in the form of a
   * SpatialObject. */
  const SpatialObjectType * GetFeature() const;

  /** Sigma value to be used in the Gaussian smoothing preceding the
   * Hessian computation. */
  itkSetMacro( Sigma, double );
  itkGetMacro( Sigma, double );

  /** Sheetness normalization value to be used in the Descoteaux sheetness filter. */
  itkSetMacro( SheetnessNormalization, double );
  itkGetMacro( SheetnessNormalization, double );

  /** Bloobiness normalization value to be used in the Descoteaux sheetness filter. */
  itkSetMacro( BloobinessNormalization, double );
  itkGetMacro( BloobinessNormalization, double );

  /** Noise normalization value to be used in the Descoteaux sheetness filter. */
  itkSetMacro( NoiseNormalization, double );
  itkGetMacro( NoiseNormalization, double );

protected:
  FrangiTubularnessFeatureGenerator();
  virtual ~FrangiTubularnessFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  FrangiTubularnessFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef InternalPixelType                           OutputPixelType;
  typedef InternalImageType                           OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef HessianRecursiveGaussianImageFilter< InputImageType >     HessianFilterType;
  typedef typename HessianFilterType::OutputImageType               HessianImageType;
  typedef typename HessianImageType::PixelType                      HessianPixelType;

  typedef  FixedArray< double, HessianPixelType::Dimension >   EigenValueArrayType;
  typedef  Image< EigenValueArrayType, Dimension >             EigenValueImageType;

  typedef  SymmetricEigenAnalysisImageFilter< HessianImageType, EigenValueImageType >     EigenAnalysisFilterType;
 
  typedef  FrangiTubularnessImageFilter< EigenValueImageType, OutputImageType >         SheetnessFilterType;

  typename HessianFilterType::Pointer             m_HessianFilter;
  typename EigenAnalysisFilterType::Pointer       m_EigenAnalysisFilter;
  typename SheetnessFilterType::Pointer           m_SheetnessFilter;

  double      m_Sigma;
  double      m_SheetnessNormalization;
  double      m_BloobinessNormalization;
  double      m_NoiseNormalization;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkFrangiTubularnessFeatureGenerator.hxx"
#endif

#endif
