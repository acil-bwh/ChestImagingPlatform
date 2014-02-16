/*=========================================================================
*
* Copyright Marius Staring, Stefan Klein, David Doria. 2011.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0.txt
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*=========================================================================*/
#ifndef __itkGaussianEnhancementImageFilter_h
#define __itkGaussianEnhancementImageFilter_h

// ITK include files
#include "itkImageToImageFilter.h"

#include "itkUnaryFunctorBase.h"
#include "itkBinaryFunctorBase.h"
#include "itkUnaryFunctorImageFilter2.h"
#include "itkBinaryFunctorImageFilter2.h"

#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

namespace itk
{

/** \class GaussianEnhancementImageFilter
 * \brief A filter to enhance image structures using Hessian
 *     measures in a single scale framework.
 *
 * \ingroup IntensityImageFilters Singlethreaded
 * \authors Changyan Xiao, Marius Staring, Denis Shamonin,
 * Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
 */

template < typename TInputImage, typename TOutputImage >
class GaussianEnhancementImageFilter
  : public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef GaussianEnhancementImageFilter                    Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage >   Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Run-time type information (and related methods) */
  itkTypeMacro( GaussianEnhancementImageFilter, ImageToImageFilter );

  /** Method for creation through the object factory.*/
  itkNewMacro( Self );

  /** Typedef's. */
  typedef TInputImage                               InputImageType;
  typedef TOutputImage                              OutputImageType;
  typedef typename InputImageType::PixelType        InputPixelType;
  typedef typename OutputImageType::PixelType       OutputPixelType;

  typedef typename NumericTraits<OutputPixelType>::RealType RealType;

  /** Image dimension = 3. */
  itkStaticConstMacro( ImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Gradient magnitude filter type, since e.g. the strain energy vesselness
   * is not only a function of the Hessian, but also of the first order derivatives.
   */
  typedef OutputPixelType                         GradientMagnitudePixelType;
  typedef Image< GradientMagnitudePixelType,
    itkGetStaticConstMacro( ImageDimension ) >    GradientMagnitudeImageType;
  typedef GradientMagnitudeRecursiveGaussianImageFilter<
    InputImageType, GradientMagnitudeImageType >  GradientMagnitudeFilterType;

  /** Hessian filter type */
  typedef Image<SymmetricSecondRankTensor<
    OutputPixelType,
    itkGetStaticConstMacro( ImageDimension ) >,
    itkGetStaticConstMacro( ImageDimension ) >    HessianTensorImageType;
  typedef HessianRecursiveGaussianImageFilter<
    InputImageType, HessianTensorImageType >      HessianFilterType;

  /** EigenValue analysis filter */
  typedef FixedArray< OutputPixelType,
    itkGetStaticConstMacro( ImageDimension ) >    EigenValueArrayType;
  typedef Image< EigenValueArrayType,
    itkGetStaticConstMacro( ImageDimension ) >    EigenValueImageType;
  typedef SymmetricEigenAnalysisImageFilter<
    HessianTensorImageType, EigenValueImageType > EigenAnalysisFilterType;

  /** Rescale filter type */
  typedef RescaleIntensityImageFilter<
    OutputImageType, OutputImageType >            RescaleFilterType;

  /** Unary functor filter type */
  typedef UnaryFunctorImageFilter2<
    EigenValueImageType,
    OutputImageType >                             UnaryFunctorImageFilterType;
  typedef typename UnaryFunctorImageFilterType::FunctorType UnaryFunctorBaseType;

  /** Binary functor filter type */
  typedef BinaryFunctorImageFilter2<
    GradientMagnitudeImageType,
    EigenValueImageType,
    OutputImageType >                             BinaryFunctorImageFilterType;
  typedef typename BinaryFunctorImageFilterType::FunctorType BinaryFunctorBaseType;

  /** Set/Get unary functor filter */
  virtual void SetUnaryFunctor( UnaryFunctorBaseType * _arg );
  itkGetObjectMacro( UnaryFunctor, UnaryFunctorBaseType );

  /** Set/Get binary functor filter */
  virtual void SetBinaryFunctor( BinaryFunctorBaseType * _arg );
  itkGetObjectMacro( BinaryFunctor, BinaryFunctorBaseType );

  /** Set/Get macros for Sigma. The current scale used.
   * Sigma should be positive. */
  itkSetClampMacro( Sigma, double, 0.0, NumericTraits<double>::max() );
  itkGetConstReferenceMacro( Sigma, double );

  /** Methods to turn on/off flag to rescale function output */
  itkSetMacro( Rescale, bool );
  itkGetConstMacro( Rescale, bool );
  itkBooleanMacro( Rescale );

  /** Define whether or not normalization factor will be used for the Gaussian. default true */
  void SetNormalizeAcrossScale( bool normalize );
  itkGetConstMacro( NormalizeAcrossScale, bool );

  /** Set the number of threads to create when executing. */
  void SetNumberOfThreads( ThreadIdType nt );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  // todo check imdim == 3
  itkConceptMacro(OutputIsFloatingPointCheck,
    (Concept::IsFloatingPoint<OutputPixelType>));
  itkConceptMacro(InputConvertibleToOutputCheck,
    (Concept::Convertible<InputPixelType, OutputPixelType>));
  /** End concept checking */
#endif

protected:
  GaussianEnhancementImageFilter();
  virtual ~GaussianEnhancementImageFilter() {};

  virtual void PrintSelf(std::ostream& os, Indent indent) const;
  virtual void GenerateData( void );

private:
  GaussianEnhancementImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Member variables. */
  typename GradientMagnitudeFilterType::Pointer   m_GradientMagnitudeFilter;
  typename HessianFilterType::Pointer             m_HessianFilter;
  typename EigenAnalysisFilterType::Pointer       m_SymmetricEigenValueFilter;
  typename RescaleFilterType::Pointer             m_RescaleFilter;

  typename UnaryFunctorBaseType::Pointer m_UnaryFunctor;
  typename BinaryFunctorBaseType::Pointer m_BinaryFunctor;
  typename UnaryFunctorImageFilterType::Pointer   m_UnaryFunctorFilter;
  typename BinaryFunctorImageFilterType::Pointer  m_BinaryFunctorFilter;

  double  m_Sigma;
  bool    m_Rescale;
  bool    m_NormalizeAcrossScale; // Normalize the image across scale space
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGaussianEnhancementImageFilter.hxx"
#endif

#endif
