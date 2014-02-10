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
#ifndef __itkFrangiXiaoSheetnessImageFilter_h
#define __itkFrangiXiaoSheetnessImageFilter_h

#include "itkBinaryFunctorImageFilter2.h"
#include "itkFrangiXiaoSheetnessFunctor.h"


namespace itk
{
/** \class FrangiXiaoSheetnessImageFilter
 * \brief Computes a measure of vesselness from the Hessian eigenvalues
 *        and the gradient magnitude.
 *
 * Based on the "Vesselness" measure proposed by Changyan Xiao et. al.
 * and on Frangi's vesselness measure.
 * The sheetness filter obtained directly from the original paper of Alejandro F. Frangi:
 * The sheetness equation is modified from eq.(13) by modifying the RA term and adding a
 * step-edge suppressing term.
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
 * \authors Alejandro F. Frangi, Wiro J. Niessen, Koen L. Vincken, Max A. Viergever
 *
 * \par Reference
 * Multiscale Vessel Enhancement Filtering.
 * Medical Image Computing and Computer-Assisted Interventation MICCAI’98
 * Lecture Notes in Computer Science, 1998, Volume 1496/1998, 130-137,
 * DOI: 10.1007/BFb0056195
 *
 * \sa FrangiVesselnessImageFilter
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInputImage1, class TInputImage2, class TOutputImage >
class ITK_EXPORT FrangiXiaoSheetnessImageFilter :
  public BinaryFunctorImageFilter2< TInputImage1, TInputImage2, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef FrangiXiaoSheetnessImageFilter        Self;
  typedef BinaryFunctorImageFilter2<
    TInputImage1, TInputImage2, TOutputImage >  Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Runtime information support. */
  itkTypeMacro( FrangiXiaoSheetnessImageFilter, BinaryFunctorImageFilter2 );

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
  typedef FrangiXiaoSheetnessFunctor<
    Input1ImagePixelType, Input2ImagePixelType,
    OutputImagePixelType >                            ThisFunctorType;

  /** Set the normalization term for . */
  void SetAlpha( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter alpha cannot be negative" );
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
  void SetC( const double & _value )
  {
    if( _value < 0.0 )
    {
      itkExceptionMacro( << "ERROR: Parameter C cannot be negative" );
    }
    this->GetFunctor().SetC( _value );
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
  FrangiXiaoSheetnessImageFilter()
  {
    /** Create and set this functor. */
    typename ThisFunctorType::Pointer thisFunctor
      = ThisFunctorType::New();
    this->SetFunctor( thisFunctor );
  }
  virtual ~FrangiXiaoSheetnessImageFilter() {}

private:
  FrangiXiaoSheetnessImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);                    // purposely not implemented

}; // end class FrangiXiaoSheetnessImageFilter

} // end namespace itk

#endif
