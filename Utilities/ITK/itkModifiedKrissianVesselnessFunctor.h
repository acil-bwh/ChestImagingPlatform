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
#ifndef __itkModifiedKrissianVesselnessFunctor_h
#define __itkModifiedKrissianVesselnessFunctor_h

#include "itkUnaryFunctorBase.h"
#include "itkComparisonOperators.h"
#include "vnl/vnl_math.h"

namespace itk
{
namespace Functor
{

/** \class ModifiedKrissianVesselnessFunctor
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

template< class TInput, class TOutput >
class ModifiedKrissianVesselnessFunctor
  : public UnaryFunctorBase< TInput, TOutput >
{
public:
  /** Standard class typedefs. */
  typedef ModifiedKrissianVesselnessFunctor   Self;
  typedef UnaryFunctorBase< TInput, TOutput > Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ModifiedKrissianVesselnessFunctor, UnaryFunctorBase );

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

    /** Compute vesselness measure. */
    const RealType vesselness = ( l2 / l3 ) * ( l2 + l3 );

    return static_cast<TOutput>( vesselness );
  } // end operator ()

  /** Set parameters */
  itkSetMacro( BrightObject, bool );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( DimensionIs3Check,
    ( Concept::SameDimension< EigenValueArrayType::Dimension, 3 > ) );
  /** End concept checking */
#endif

protected:
  /** Constructor */
  ModifiedKrissianVesselnessFunctor()
  {
    this->m_BrightObject = true;
  };
  virtual ~ModifiedKrissianVesselnessFunctor(){};

private:
  ModifiedKrissianVesselnessFunctor(const Self &);  // purposely not implemented
  void operator=(const Self &);                     // purposely not implemented

  /** Member variables. */
  bool    m_BrightObject;

}; // end class ModifiedKrissianVesselnessFunctor

} // end namespace itk::Functor
} // end namespace itk

#endif
