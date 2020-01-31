/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkDescoteauxSheetnessImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDescoteauxSheetnessImageFilter_h
#define __itkDescoteauxSheetnessImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{
  
/** \class DescoteauxSheetnessImageFilter
 *
 * \brief Computes a measure of Sheetness from the Hessian Eigenvalues
 *
 * Based on the "Sheetness" measure proposed by Decouteaux et. al.
 * 
 * M.Descoteaux, M.Audette, K.Chinzei, el al.: 
 * "Bone enhancement filtering: Application to sinus bone segmentation
 *  and simulation of pituitary surgery."
 *  In: MICCAI.  (2005) 9–16
 *
 * \ingroup IntensityImageFilters  Multithreaded
 * \ingroup LesionSizingToolkit
 */
namespace Function {  
  
template< class TInput, class TOutput>
class Sheetness
{
public:
  Sheetness() 
    {
    m_Alpha = 0.5; // suggested value in the paper
    m_Gamma = 0.5; // suggested value in the paper;
    m_C     = 1.0;
    m_DetectBrightSheets = true;
    }
  ~Sheetness() {}
  bool operator!=( const Sheetness & ) const
    {
    return false;
    }
  bool operator==( const Sheetness & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A )
    {
    double sheetness = 0.0;

    double a1 = static_cast<double>( A[0] );
    double a2 = static_cast<double>( A[1] );
    double a3 = static_cast<double>( A[2] );

    double l1 = vnl_math_abs( a1 );
    double l2 = vnl_math_abs( a2 );
    double l3 = vnl_math_abs( a3 );

    //
    // Sort the values by their absolute value.
    // At the end of the sorting we should have
    // 
    //          l1 <= l2 <= l3
    //
    if( l2 > l3 )
      {
      double tmpl = l3;
      l3 = l2;
      l2 = tmpl;
      double tmpa = a3;
      a3 = a2;
      a2 = tmpa;
      }

    if( l1 > l2 )
      {
      double tmp = l1;
      l1 = l2;
      l2 = tmp;
      double tmpa = a1;
      a1 = a2;
      a2 = tmpa;
      }   

    if( l2 > l3 )
      {
      double tmp = l3;
      l3 = l2;
      l2 = tmp;
      double tmpa = a3;
      a3 = a2;
      a2 = tmpa;
      }
    
    if( this->m_DetectBrightSheets )
      {
      if( a3 > 0.0 ) 
        {
        return static_cast<TOutput>( sheetness );
        }
      }
    else
      {
      if( a3 < 0.0 ) 
        {
        return static_cast<TOutput>( sheetness );
        }
      }


    //
    // Avoid divisions by zero (or close to zero)
    //
    if( static_cast<double>( l3 ) < vnl_math::eps )
      {
      return static_cast<TOutput>( sheetness );
      } 

    const double Rs = l2 / l3;
    const double Rb = vnl_math_abs( l3 + l3 - l2 - l1 ) / l3;
    const double Rn = std::sqrt( l3*l3 + l2*l2 + l1*l1 );

    sheetness  =         std::exp( - ( Rs * Rs ) / ( 2.0 * m_Alpha * m_Alpha ) ); 
    sheetness *= ( 1.0 - std::exp( - ( Rb * Rb ) / ( 2.0 * m_Gamma * m_Gamma ) ) ); 
    sheetness *= ( 1.0 - std::exp( - ( Rn * Rn ) / ( 2.0 * m_C     * m_C     ) ) ); 

    return static_cast<TOutput>( sheetness );
    }
  void SetAlpha( double value )
    {
    this->m_Alpha = value;
    }
  void SetGamma( double value )
    {
    this->m_Gamma = value;
    }
  void SetC( double value )
    {
    this->m_C = value;
    }
  void SetDetectBrightSheets( bool value )
    {
    this->m_DetectBrightSheets = value;
    }
  void SetDetectDarkSheets( bool value )
    {
    this->m_DetectBrightSheets = !value;
    }

private:
  double    m_Alpha;
  double    m_Gamma;
  double    m_C;
  bool      m_DetectBrightSheets;
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT DescoteauxSheetnessImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::Sheetness< typename TInputImage::PixelType, 
                                       typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef DescoteauxSheetnessImageFilter    Self;
  typedef UnaryFunctorImageFilter<
    TInputImage,TOutputImage, 
    Function::Sheetness< 
      typename TInputImage::PixelType, 
      typename TOutputImage::PixelType> >   Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(DescoteauxSheetnessImageFilter, 
               UnaryFunctorImageFilter);

  /** Set the normalization term for sheetness */
  void SetSheetnessNormalization( double value )
    {
    this->GetFunctor().SetAlpha( value );
    }

  /** Set the normalization term for bloobiness. */
  void SetBloobinessNormalization( double value )
    {
    this->GetFunctor().SetGamma( value );
    }

  /** Set the normalization term for noise. */
  void SetNoiseNormalization( double value )
    {
    this->GetFunctor().SetC( value );
    }
  void SetDetectBrightSheets( bool value )
    {
    this->GetFunctor().SetDetectBrightSheets( value );
    }
  void SetDetectDarkSheets( bool value )
    {
    this->GetFunctor().SetDetectDarkSheets( value );
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
  DescoteauxSheetnessImageFilter() {}
  virtual ~DescoteauxSheetnessImageFilter() {}

private:
  DescoteauxSheetnessImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
