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
#ifndef __itkFrangiVesselnessFunctor_h
#define __itkFrangiVesselnessFunctor_h

#include "itkUnaryFunctorBase.h"
#include "itkComparisonOperators.h"
#include "vnl/vnl_math.h"

namespace itk
{
namespace Functor
{

/** \class FrangiVesselnessFunctor
 * \brief Computes a measure of vesselness from the Hessian eigenvalues.
 *
 * Based on the "Vesselness" measure proposed by Alejandro F. Frangi et. al.
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

template< class TInput, class TOutput >
class FrangiVesselnessFunctor
  : public UnaryFunctorBase< TInput, TOutput >
{
public:
  /** Standard class typedefs. */
  typedef FrangiVesselnessFunctor             Self;
  typedef UnaryFunctorBase< TInput, TOutput > Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( FrangiVesselnessFunctor, UnaryFunctorBase );

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
    if( l2 < vnl_math::eps || l3 < vnl_math::eps )
    {
      return NumericTraits<TOutput>::Zero;
    }

    /** Compute several structure measures. */
    const RealType Ra = l2 / l3; // see Eq.(11)
    const RealType Rb = l1 / vcl_sqrt( l2 * l3 ); // see Eq.(10)
    const RealType S  = vcl_sqrt( l1 * l1 + l2 * l2 + l3 * l3 ); // see Eq.(12)

    /** Compute final vesselness measure, see Eq.(13). */
    RealType vesselness = NumericTraits<RealType>::Zero;
    vesselness  = ( 1.0 - vcl_exp( -( Ra * Ra ) / ( 2.0 * m_Alpha * m_Alpha ) ) ); // sheetness vs lineness
    vesselness *=         vcl_exp( -( Rb * Rb ) / ( 2.0 * m_Beta  * m_Beta  ) );   // blobness
    vesselness *= ( 1.0 - vcl_exp( -( S  * S )  / ( 2.0 * m_C     * m_C ) ) );     // noise = structuredness

    return static_cast<TOutput>( vesselness );
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
  FrangiVesselnessFunctor()
  {
    this->m_Alpha = 0.5; // suggested value in the paper
    this->m_Beta = 0.5;  // suggested value in the paper
    this->m_C = 500.0;   // Depends on intensity range, 500 good for lung CT
    this->m_BrightObject = true;
  };
  virtual ~FrangiVesselnessFunctor(){};

private:
  FrangiVesselnessFunctor(const Self &); // purposely not implemented
  void operator=(const Self &);          // purposely not implemented

  /** Member variables. */
  double  m_Alpha;
  double  m_Beta;
  double  m_C;
  bool    m_BrightObject;

}; // end class FrangiVesselnessFunctor

} // end namespace itk::Functor
} // end namespace itk

#endif
