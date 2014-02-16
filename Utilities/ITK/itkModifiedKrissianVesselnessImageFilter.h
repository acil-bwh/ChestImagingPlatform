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
#ifndef __itkModifiedKrissianVesselnessImageFilter_h
#define __itkModifiedKrissianVesselnessImageFilter_h

#include "itkUnaryFunctorImageFilter2.h"
#include "itkModifiedKrissianVesselnessFunctor.h"


namespace itk
{
/** \class ModifiedKrissianVesselnessImageFilter
 * \brief Computes a measure of vesselness from the Hessian eigenvalues.
 *
 * Inspired by the paper:
 * \authors Krissian, K. and Malandain, G. and Ayache, N. and Vaillant, R. and Trousset, Y.
 *
 * \par Reference
 * Model Based Detection of Tubular Structures in 3D Images
 * Computer Vision and Image Understanding, vol. 80, no. 2, pp. 130 - 171, Nov. 2000.
 *
 * \sa FrangiVesselnessImageFilter
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInputImage, class TOutputImage >
class ITK_EXPORT ModifiedKrissianVesselnessImageFilter :
  public UnaryFunctorImageFilter2< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef ModifiedKrissianVesselnessImageFilter       Self;
  typedef UnaryFunctorImageFilter2<
    TInputImage, TOutputImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Runtime information support. */
  itkTypeMacro( ModifiedKrissianVesselnessImageFilter, UnaryFunctorImageFilter2 );

  /** Some typedefs. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageRegionType   InputImageRegionType;
  typedef typename Superclass::InputImagePixelType    InputImagePixelType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::OutputImageRegionType  OutputImageRegionType;
  typedef typename Superclass::OutputImagePixelType   OutputImagePixelType;
  typedef ModifiedKrissianVesselnessFunctor<
    InputImagePixelType, OutputImagePixelType >       ThisFunctorType;

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
  ModifiedKrissianVesselnessImageFilter()
  {
    /** Create and set this functor. */
    typename ThisFunctorType::Pointer thisFunctor
      = ThisFunctorType::New();
    this->SetFunctor( thisFunctor );
  }
  virtual ~ModifiedKrissianVesselnessImageFilter() {}

private:
  ModifiedKrissianVesselnessImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);              // purposely not implemented

}; // end class ModifiedKrissianVesselnessImageFilter

} // end namespace itk

#endif
