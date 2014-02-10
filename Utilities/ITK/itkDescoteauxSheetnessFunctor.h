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
#ifndef __itkDescoteauxSheetnessFunctor_h
#define __itkDescoteauxSheetnessFunctor_h

#include "itkUnaryFunctorBase.h"
#include "itkComparisonOperators.h"
#include "vnl/vnl_math.h"

namespace itk
{
namespace Functor
{

/** \class DescoteauxSheetnessFunctor
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
 * \sa FrangiVesselnessFunctor
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInput, class TOutput >
class DescoteauxSheetnessFunctor
  : public UnaryFunctorBase< TInput, TOutput >
{
public:
  /** Standard class typedefs. */
  typedef DescoteauxSheetnessFunctor          Self;
  typedef UnaryFunctorBase< TInput, TOutput > Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( DescoteauxSheetnessFunctor, UnaryFunctorBase );

  /** Typedef's. */
  typedef typename NumericTraits<TOutput>::RealType RealType;
  typedef TInput                                    EigenValueArrayType;
  typedef typename EigenValueArrayType::ValueType   EigenValueType;

  /** This does the real computation */
  virtual TOutput Evaluate( const TInput & eigenValues ) const
  {
    /** Sort the eigenvalues by their absolute value, such that |l1| < |l2| < |l3|. */
    EigenValueArrayType sortedEigenValues = eigenValues;
    std::sort( sortedEigenValues.Begin(), sortedEigenValues.End(),
      Functor::AbsLessCompare<EigenValueType>() );

    /** Take the absolute values and abbreviate. */
    const RealType l1 = vnl_math_abs( sortedEigenValues[ 0 ] );
    const RealType l2 = vnl_math_abs( sortedEigenValues[ 1 ] );
    const RealType l3 = vnl_math_abs( sortedEigenValues[ 2 ] );

    /** Reject. */
    if( this->m_BrightObject )
    {
      // Reject dark tubes and dark ridges over bright background
      if( sortedEigenValues[ 2 ] > NumericTraits<RealType>::Zero )
      {
        return NumericTraits<TOutput>::Zero;
      }
    }
    else
    {
      // Reject bright tubes and bright ridges over dark background
      if( sortedEigenValues[ 2 ] < NumericTraits<RealType>::Zero )
      {
        return NumericTraits<TOutput>::Zero;
      }
    }

    /** Avoid divisions by zero (or close to zero). */
    if( l3 < vnl_math::eps )
    {
      return NumericTraits<TOutput>::Zero;
    }

    /** Compute several structure measures. */
    const RealType Rsheet = l2 / l3;
    const RealType Rblob  = vnl_math_abs( l3 + l3 - l2 - l1 ) / l3;
    const RealType Rnoise = vcl_sqrt( l1 * l1 + l2 * l2 + l3 * l3 );

    /** Compute final sheetness measure, see Eq.(13). */
    RealType sheetness = NumericTraits<RealType>::Zero;
    sheetness  =         vcl_exp( - ( Rsheet * Rsheet ) / ( 2.0 * this->m_Alpha * this->m_Alpha ) ); // sheetness vs lineness
    sheetness *= ( 1.0 - vcl_exp( - ( Rblob  * Rblob )  / ( 2.0 * this->m_Beta * this->m_Beta ) ) ); // blobness
    sheetness *= ( 1.0 - vcl_exp( - ( Rnoise * Rnoise ) / ( 2.0 * this->m_C * this->m_C ) ) );       // noise = structuredness

    return static_cast<TOutput>( sheetness );
  } // end operator ()

  /** Set parameters */
  itkSetClampMacro( Alpha, double, 0.0, NumericTraits<double>::max() );
  itkSetClampMacro( Beta, double, 0.0, NumericTraits<double>::max() );
  itkSetClampMacro( C, double, 0.0, NumericTraits<double>::max() );
  itkSetMacro( BrightObject, bool );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( DimensionIs3Check,
    ( Concept::SameDimension< EigenValueArrayType::Dimension, 3 > ) );
  /** End concept checking */
#endif

protected:
  /** Constructor */
  DescoteauxSheetnessFunctor()
  {
    this->m_Alpha = 0.5; // suggested value in the paper
    this->m_Beta = 0.5;  // suggested value in the paper
    this->m_C = 1.0;     // good for lung CT
    this->m_BrightObject = true;
  };
  virtual ~DescoteauxSheetnessFunctor(){};

private:
  DescoteauxSheetnessFunctor(const Self &);  //purposely not implemented
  void operator=(const Self &); //purposely not implemented

  /** Member variables. */
  double  m_Alpha;
  double  m_Beta;
  double  m_C;
  bool    m_BrightObject;

}; // end class DescoteauxSheetnessFunctor

} // end namespace itk::Functor
} // end namespace itk

#endif
