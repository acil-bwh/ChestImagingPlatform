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
#ifndef __itkDescoteauxSheetnessImageFilter_h
#define __itkDescoteauxSheetnessImageFilter_h

#include "itkUnaryFunctorImageFilter2.h"
#include "itkDescoteauxSheetnessFunctor.h"


namespace itk
{
/** \class DescoteauxSheetnessImageFilter
 * \brief Computes a measure of sheetness from the Hessian eigenvalues.
 *
 * Based on the "Sheetness" measure proposed by M. Descoteaux et. al.
 *
 * \authors M. Descoteaux, M. Audette, K. Chinzei, K. Siddiqi
 *
 * \par Reference
 * Bone Enhancement Filtering: Application to Sinus Bone Segmentation
 * and Simulation of Pituitary Surgery. In Proceedings of MICCAI'2005. pp.9~16
 *
 * \sa DescoteauxSheetnessFunctor
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInputImage, class TOutputImage >
class ITK_EXPORT DescoteauxSheetnessImageFilter :
  public UnaryFunctorImageFilter2< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DescoteauxSheetnessImageFilter       Self;
  typedef UnaryFunctorImageFilter2<
    TInputImage, TOutputImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Runtime information support. */
  itkTypeMacro( DescoteauxSheetnessImageFilter, UnaryFunctorImageFilter2 );

  /** Some typedefs. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageRegionType   InputImageRegionType;
  typedef typename Superclass::InputImagePixelType    InputImagePixelType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::OutputImageRegionType  OutputImageRegionType;
  typedef typename Superclass::OutputImagePixelType   OutputImagePixelType;
  typedef DescoteauxSheetnessFunctor<
    InputImagePixelType, OutputImagePixelType >       ThisFunctorType;

  /** Set the normalization term for sheetness. */
  void SetAlpha( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter alpha cannot be negative" );
    }
    this->GetFunctor().SetAlpha( _value );
    this->Modified();
  }

  /** Set the normalization term for blobness. */
  void SetBeta( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter beta cannot be negative" );
    }
    this->GetFunctor().SetBeta( _value );
    this->Modified();
  }

  /** Set the normalization term for noise. */
  void SetC( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter C cannot be negative" );
    }
    this->GetFunctor().SetC( _value );
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
  DescoteauxSheetnessImageFilter()
  {
    /** Create and set this functor. */
    typename ThisFunctorType::Pointer thisFunctor
      = ThisFunctorType::New();
    this->SetFunctor( thisFunctor );
  }
  virtual ~DescoteauxSheetnessImageFilter() {}

private:
  DescoteauxSheetnessImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);              // purposely not implemented

}; // end class DescoteauxSheetnessImageFilter

} // end namespace itk

#endif
