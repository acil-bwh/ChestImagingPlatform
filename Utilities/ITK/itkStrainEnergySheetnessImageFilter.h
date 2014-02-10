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
#ifndef __itkStrainEnergySheetnessImageFilter_h
#define __itkStrainEnergySheetnessImageFilter_h

#include "itkBinaryFunctorImageFilter2.h"
#include "itkStrainEnergySheetnessFunctor.h"


namespace itk
{
/** \class StrainEnergySheetnessImageFilter
 * \brief Computes a measure of sheetness from the Hessian eigenvalues
 *        and the gradient magnitude.
 *
 * Based on the "Vesselness" measure proposed by Changyan Xiao et. al.
 *
 * \authors Changyan Xiao, Marius Staring, Denis Shamonin,
 * Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
 *
 * \par Reference
 * A strain energy filter for 3D vessel enhancement with application to pulmonary CT images,
 * Medical Image Analysis, Volume 15, Issue 1, February 2011, Pages 112-124,
 * ISSN 1361-8415, DOI: 10.1016/j.media.2010.08.003.
 * http://www.sciencedirect.com/science/article/B6W6Y-5137FFY-1/2/8238fdff2ee2a26858b794913bce6546
 *
 * \sa FrangiVesselnessImageFilter
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInputImage1, class TInputImage2, class TOutputImage >
class ITK_EXPORT StrainEnergySheetnessImageFilter :
  public BinaryFunctorImageFilter2< TInputImage1, TInputImage2, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef StrainEnergySheetnessImageFilter      Self;
  typedef BinaryFunctorImageFilter2<
    TInputImage1, TInputImage2, TOutputImage >  Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Runtime information support. */
  itkTypeMacro( StrainEnergySheetnessImageFilter, BinaryFunctorImageFilter2 );

  /** Some typedefs. */
  typedef typename Superclass::Input1ImageType        Input1ImageType;
  typedef typename Superclass::Input1ImagePointer     Input1ImagePointer;
  typedef typename Superclass::Input1ImageRegionType  Input1ImageRegionType;
  typedef typename Superclass::Input1ImagePixelType   Input1ImagePixelType;
  typedef typename Superclass::Input2ImageType        Input2ImageType;
  typedef typename Superclass::Input2ImagePointer     Input2ImagePointer;
  typedef typename Superclass::Input2ImageRegionType  Input2ImageRegionType;
  typedef typename Superclass::Input2ImagePixelType   Input2ImagePixelType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::OutputImageRegionType  OutputImageRegionType;
  typedef typename Superclass::OutputImagePixelType   OutputImagePixelType;
  typedef StrainEnergySheetnessFunctor<
    Input1ImagePixelType, Input2ImagePixelType,
    OutputImagePixelType >                            ThisFunctorType;

  /** Set the normalization term for . */
  void SetAlpha( const double & _value )
  {
    if( _value < 0.0 || _value > 1.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter alpha cannot be negative or greater than 1" );
    }
    this->GetFunctor().SetAlpha( _value );
    this->Modified();
  }

  /** Set the normalization term for . */
  void SetBeta( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter beta cannot be negative" );
    }
    this->GetFunctor().SetBeta( _value );
    this->Modified();
  }

  /** Set the normalization term for . */
  void SetNu( const double & _value )
  {
    if( _value < -1.0 || _value > 0.5 )
    {
      itkExceptionMacro( << "ERROR: Parameter nu should be in range [-1, 0.5]" );
    }
    this->GetFunctor().SetNu( _value );
    this->Modified();
  }

  /** Set the normalization term for . */
  void SetKappa( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter kappa cannot be negative" );
    }
    this->GetFunctor().SetKappa( _value );
    this->Modified();
  }

  /** Enhance bright structures on a dark background if true,
   * the opposite if false. Default true.
   */
  void SetBrightObject( const bool & _value )
  {
    this->GetFunctor().SetBrightObject( _value );
  }

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  typedef typename TInputImage::PixelType InputPixelType;
  itkConceptMacro(BracketOperatorsCheck,
    (Concept::BracketOperator< InputPixelType, unsigned int, double >));
  itkConceptMacro(DoubleConvertibleToOutputCheck,
    (Concept::Convertible<double, typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  StrainEnergySheetnessImageFilter()
  {
    /** Create and set this functor. */
    typename ThisFunctorType::Pointer thisFunctor
      = ThisFunctorType::New();
    this->SetFunctor( thisFunctor );
  }
  virtual ~StrainEnergySheetnessImageFilter() {}

private:
  StrainEnergySheetnessImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);                   // purposely not implemented

}; // end class StrainEnergySheetnessImageFilter

} // end namespace itk

#endif
