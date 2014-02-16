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
#ifndef __itkFrangiSheetnessImageFilter_h
#define __itkFrangiSheetnessImageFilter_h

#include "itkUnaryFunctorImageFilter2.h"
#include "itkFrangiSheetnessFunctor.h"


namespace itk
{

/** \class FrangiSheetnessImageFilter
 * \brief Computes a measure of sheetness from the Hessian eigenvalues.
 *
 * The sheetness filter obtained directly from the original paper
 * of Alejandro F. Frangi by modifying the RA term.
 *
 * \authors Alejandro F. Frangi, Wiro J. Niessen, Koen L. Vincken, Max A. Viergever
 *
 * \par Reference
 * Multiscale Vessel Enhancement Filtering.
 * Medical Image Computing and Computer-Assisted Interventation MICCAI’98
 * Lecture Notes in Computer Science, 1998, Volume 1496/1998, 130-137,
 * DOI: 10.1007/BFb0056195
 *
 * \sa UnaryFunctorImageFilter
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInputImage, class TOutputImage >
class ITK_EXPORT FrangiSheetnessImageFilter :
  public UnaryFunctorImageFilter2< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef FrangiSheetnessImageFilter       Self;
  typedef UnaryFunctorImageFilter2<
    TInputImage, TOutputImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Runtime information support. */
  itkTypeMacro( FrangiSheetnessImageFilter, UnaryFunctorImageFilter2 );

  /** Some typedefs. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageRegionType   InputImageRegionType;
  typedef typename Superclass::InputImagePixelType    InputImagePixelType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::OutputImageRegionType  OutputImageRegionType;
  typedef typename Superclass::OutputImagePixelType   OutputImagePixelType;
  typedef FrangiSheetnessFunctor<
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
  FrangiSheetnessImageFilter()
  {
    /** Create and set this functor. */
    typename ThisFunctorType::Pointer thisFunctor
      = ThisFunctorType::New();
    this->SetFunctor( thisFunctor );
  }
  virtual ~FrangiSheetnessImageFilter() {}

private:
  FrangiSheetnessImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);              // purposely not implemented

}; // end class FrangiSheetnessImageFilter

} // end namespace itk

#endif
