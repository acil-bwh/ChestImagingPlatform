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
#ifndef __itkMultiScaleGaussianEnhancementImageFilter_h
#define __itkMultiScaleGaussianEnhancementImageFilter_h

#include "itkGaussianEnhancementImageFilter.h"

namespace itk
{
/**\class MultiScaleGaussianEnhancementImageFilter
 * \brief A filter to enhance image structures using Hessian
 * measures in a multi scale framework.
 *
 * Minimum and maximum sigma value can be set using SetSigmaMinimum
 * and SetSigmaMaximum methods respectively. The number of scale levels is set
 * using SetNumberOfSigmaSteps method. Exponentially distributed scale levels are
 * computed within the bound set by the minimum and maximum sigma values.
 *
 * The filter computes a second output image (accessed by the GetScalesOutput method)
 * containing the scales at which each pixel gave the best response.
 *
 * \sa GaussianEnhancementImageFilter
 * \sa HessianRecursiveGaussianImageFilter
 * \sa SymmetricEigenAnalysisImageFilter
 * \sa DescoteauxSheetnessImageFilter
 * \sa FrangiSheetnessImageFilter
 * \sa StrainEnergyVesselnessImageFilter
 *
 * \ingroup IntensityImageFilters TensorObjects
 * \authors Changyan Xiao, Marius Staring, Denis Shamonin,
 * Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
 */
template < typename TInputImage, typename TOutputImage >
class MultiScaleGaussianEnhancementImageFilter
  : public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MultiScaleGaussianEnhancementImageFilter        Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  typedef TInputImage                                     InputImageType;
  typedef TOutputImage                                    OutputImageType;
  typedef typename TInputImage::PixelType                 InputPixelType;
  typedef typename TOutputImage::PixelType                OutputPixelType;
  typedef typename TOutputImage::RegionType               OutputRegionType;

  /** DataObject pointer */
  typedef DataObject::Pointer                             DataObjectPointer;

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiScaleGaussianEnhancementImageFilter, ImageToImageFilter );

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Image dimension. */
  itkStaticConstMacro( ImageDimension, unsigned int,
    InputImageType::ImageDimension );

  /** Types for Scales image */
  typedef OutputPixelType                                 ScalesPixelType;
  typedef Image< ScalesPixelType,
    itkGetStaticConstMacro( ImageDimension ) >            ScalesImageType;

  /** Single scale Gaussian enhancement filter type. */
  typedef GaussianEnhancementImageFilter<
    InputImageType, OutputImageType >                     SingleScaleFilterType;
  typedef typename SingleScaleFilterType::GradientMagnitudePixelType    GradientMagnitudePixelType;
  typedef typename SingleScaleFilterType::GradientMagnitudeImageType    GradientMagnitudeImageType;
  typedef typename SingleScaleFilterType::GradientMagnitudeFilterType   GradientMagnitudeFilterType;
  typedef typename SingleScaleFilterType::HessianTensorImageType        HessianTensorImageType;
  typedef typename SingleScaleFilterType::HessianFilterType             HessianFilterType;
  typedef typename SingleScaleFilterType::EigenValueArrayType           EigenValueArrayType;
  typedef typename SingleScaleFilterType::EigenValueImageType           EigenValueImageType;
  typedef typename SingleScaleFilterType::EigenAnalysisFilterType       EigenAnalysisFilterType;
  typedef typename SingleScaleFilterType::RescaleFilterType             RescaleFilterType;
  typedef typename SingleScaleFilterType::UnaryFunctorImageFilterType   UnaryFunctorImageFilterType;
  typedef typename SingleScaleFilterType::UnaryFunctorBaseType          UnaryFunctorBaseType;
  typedef typename SingleScaleFilterType::BinaryFunctorImageFilterType  BinaryFunctorImageFilterType;
  typedef typename SingleScaleFilterType::BinaryFunctorBaseType         BinaryFunctorBaseType;

  /** Set/Get unary functor */
  virtual void SetUnaryFunctor( UnaryFunctorBaseType * _arg );
  //itkGetObjectMacro( UnaryFunctor, UnaryFunctorBaseType );

  /** Set/Get binary functor */
  virtual void SetBinaryFunctor( BinaryFunctorBaseType * _arg );
  //itkGetObjectMacro( BinaryFunctor, BinaryFunctorBaseType );

  /** Set/Get macros for sigma minimum */
  itkSetClampMacro( SigmaMinimum, double, 0.0, NumericTraits<double>::max() );
  itkGetConstMacro( SigmaMinimum, double );

  /** Set/Get macros for sigma maximum */
  itkSetClampMacro( SigmaMaximum, double, 0.0, NumericTraits<double>::max() );
  itkGetConstMacro( SigmaMaximum, double );

  /** Set/Get macros for number of scales */
  itkSetClampMacro( NumberOfSigmaSteps, unsigned int, 1, NumericTraits<unsigned int>::max() );
  itkGetConstMacro( NumberOfSigmaSteps, unsigned int );

  /** Methods to turn on/off flag to inform the filter that the Hessian-based
   * measure is non-negative (classical measures like Sato's and Frangi's are),
   * hence it has a minimum at zero. In this case, the update buffer is
   * initialized at zero, and the output scale and Hessian are zero in case
   * the Hessian-based measure returns zero for all scales. Otherwise, the
   * minimum output scale and Hessian are the ones obtained at scale
   * SigmaMinimum. On by default.
   */
  itkSetMacro( NonNegativeHessianBasedMeasure, bool );
  itkGetConstMacro( NonNegativeHessianBasedMeasure, bool );
  itkBooleanMacro( NonNegativeHessianBasedMeasure );

  typedef enum { EquispacedSigmaSteps = 0,
    LogarithmicSigmaSteps = 1 } SigmaStepMethodType;

  /** Set/Get the method used to generate scale sequence (Equispaced or Logarithmic) */
  itkSetMacro( SigmaStepMethod, SigmaStepMethodType );
  itkGetConstMacro( SigmaStepMethod, SigmaStepMethodType );
  virtual void SetSigmaStepMethod( unsigned int arg )
  {
    if( arg == 0 )
    {
      this->m_SigmaStepMethod = Self::EquispacedSigmaSteps;
    }
    else if( arg == 1 )
    {
      this->m_SigmaStepMethod = Self::LogarithmicSigmaSteps;
    }
    else
    {
      itkExceptionMacro( << "ERROR: this SigmaStepMethod is not supported!" );
    }
  }

  /** Set equispaced sigma step method */
  void SetSigmaStepMethodToEquispaced( void );

  /** Set logarithmic sigma step method */
  void SetSigmaStepMethodToLogarithmic( void );

  /** Methods to turn on/off flag to rescale function output */
  itkSetMacro( Rescale, bool );
  itkGetConstMacro( Rescale, bool );
  itkBooleanMacro( Rescale );

  /** Methods to turn on/off flag to generate an image with scale values at
  *  each pixel for the best vesselness response */
  itkSetMacro( GenerateScalesOutput, bool );
  itkGetConstMacro( GenerateScalesOutput, bool );
  itkBooleanMacro( GenerateScalesOutput );

  /** Define whether or not normalization factor will be used for the Gaussian. default true */
  void SetNormalizeAcrossScale( bool normalize );
  bool GetNormalizeAcrossScale() const;

  /** Set the number of threads to create when executing. */
  void SetNumberOfThreads( ThreadIdType nt );

  /** Get the image containing the scales at which each pixel gave the best response */
  const ScalesImageType * GetScalesOutput( void ) const;

  /** This is overloaded to create the Scales and Hessian output images */
  virtual DataObjectPointer MakeOutput( unsigned int idx );

  void EnlargeOutputRequestedRegion( DataObject * );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputIsFloatingPointCheck,
    (Concept::IsFloatingPoint<OutputPixelType>));
  /** End concept checking */
#endif

protected:
  MultiScaleGaussianEnhancementImageFilter();
  ~MultiScaleGaussianEnhancementImageFilter() {};

  /** Does the work. */
  virtual void GenerateData( void );

  /** Print member variables. */
  virtual void PrintSelf( std::ostream& os, Indent indent ) const;

private:
  MultiScaleGaussianEnhancementImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);                           // purposely not implemented

  /** Computes the maximum of all single scale responses. */
  void UpdateMaximumResponse(
    const OutputImageType * seOutput,
    const unsigned int & scaleLevel );

  /** Compute the current sigma. */
  double ComputeSigmaValue( const unsigned int & scaleLevel );

  /** Single scale filter */
  typename SingleScaleFilterType::Pointer m_GaussianEnhancementFilter;

  /** Member variables. */
  bool                 m_NonNegativeHessianBasedMeasure;
  bool                 m_GenerateScalesOutput;
  bool                 m_Rescale;

  double               m_SigmaMinimum;
  double               m_SigmaMaximum;
  unsigned int         m_NumberOfSigmaSteps;
  SigmaStepMethodType  m_SigmaStepMethod;

}; // end class MultiScaleGaussianEnhancementImageFilter

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiScaleGaussianEnhancementImageFilter.hxx"
#endif

#endif
