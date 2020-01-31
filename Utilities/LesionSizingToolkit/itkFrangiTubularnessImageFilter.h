/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFrangiTubularnessImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFrangiTubularnessImageFilter_h
#define __itkFrangiTubularnessImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{
  
/** \class FrangiTubularnessImageFilter
 *
 * \brief Computes a measure of CrestLines from the Hessian Eigenvalues
 *
 * Based on the "Tubularness" measure proposed by Frangi et al.
 * 
 * \ingroup IntensityImageFilters  Multithreaded
 * \ingroup LesionSizingToolkit
 */
namespace Function {  
  
template< class TInput, class TOutput>
class Tubularness
{
public:
  Tubularness() 
    {
    m_Alpha  = 0.5; // suggested value in the paper
    m_Beta   = 0.5; // suggested value in the paper;
    m_Gamma  = 1.0; // suggested value in the paper;
    m_BrigthForeground = true;
    }
  ~Tubularness() {}
  Tubularness( const Tubularness & one )
    {
    this->m_Alpha = one.m_Alpha;
    this->m_Beta = one.m_Beta;
    this->m_Gamma = one.m_Gamma;
    this->m_BrigthForeground = one.m_BrigthForeground;
    }
  const Tubularness & operator=( const Tubularness & one )
    {
    this->m_Alpha = one.m_Alpha;
    this->m_Beta = one.m_Beta;
    this->m_Gamma = one.m_Gamma;
    this->m_BrigthForeground = one.m_BrigthForeground;
    return *this;
    }
  bool operator!=( const Tubularness & ) const
    {
    return false;
    }
  bool operator==( const Tubularness & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A )
    {
    double tubularness = 0.0;
    
    double a1 = static_cast<double>( A[0] );
    double a2 = static_cast<double>( A[1] );
    double a3 = static_cast<double>( A[2] );

    double l1 = vnl_math_abs( a1 );
    double l2 = vnl_math_abs( a2 );
    double l3 = vnl_math_abs( a3 );

    //
    // Sort the values by their absolute value.
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
    
    if( m_BrigthForeground )
      {
      //
      // Reject dark tubes and dark ridges over bright background
      //
      if( a3 > 0.0 ) 
        {
        return tubularness;
        }
      }
    else
      {
      //
      // Reject bright tubes and bright ridges over dark background
      //
      if( a3 < 0.0 ) 
        {
        return tubularness;
        }
      }

    // avoid divisions by zero
    if( l2 < vnl_math::eps || l3 < vnl_math::eps )
      {
      return tubularness;
      }

    const double Rs = l2 / l3;
    const double Rb = l1 / std::sqrt( l2 * l3 );
    const double Rn = std::sqrt( l3*l3 + l2*l2 + l1*l1 );

    tubularness  = ( 1.0 - std::exp( - ( Rs * Rs ) / ( 2.0 * m_Alpha * m_Alpha ) ) ); 
    tubularness *= (       std::exp( - ( Rb * Rb ) / ( 2.0 * m_Beta  * m_Beta  ) ) ); 
    tubularness *= ( 1.0 - std::exp( - ( Rn * Rn ) / ( 2.0 * m_Gamma * m_Gamma ) ) ); 

    return static_cast<TOutput>( tubularness );
    }

  void SetAlpha( double value )
    {
    this->m_Alpha = value;
    }
  void SetBeta( double value )
    {
    this->m_Beta = value;
    }
  void SetGamma( double value )
    {
    this->m_Gamma = value;
    }
  void SetBrightBackground( bool value )
    {
    std::cout << "m_BrigthForeground " << value << std::endl;
    this->m_BrigthForeground;
    }

private:
  double m_Alpha;
  double m_Beta;
  double m_Gamma;
  bool   m_BrigthForeground;
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT FrangiTubularnessImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::Tubularness< typename TInputImage::PixelType, 
                                       typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef FrangiTubularnessImageFilter          Self;
  typedef UnaryFunctorImageFilter<
    TInputImage,TOutputImage, 
    Function::Tubularness< 
      typename TInputImage::PixelType, 
      typename TOutputImage::PixelType> >       Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FrangiTubularnessImageFilter, 
               UnaryFunctorImageFilter);

  /** Set the normalization term for sheetness */
  void SetSheetnessNormalization( double value )
    {
    this->GetFunctor().SetAlpha( value );
    }

  /** Set the normalization term for bloobiness. */
  void SetBloobinessNormalization( double value )
    {
    this->GetFunctor().SetBeta( value );
    }

  /** Set the normalization term for noise. */
  void SetNoiseNormalization( double value )
    {
    this->GetFunctor().SetGamma( value );
    }

  /** Pick whether to search for brigh tubes over dark background
   *  or dark tubes over bright background. */
  void SetBrightBackground( bool value )
    {
    this->GetFunctor().SetBrightBackground( value );
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
  FrangiTubularnessImageFilter() {}
  virtual ~FrangiTubularnessImageFilter() {}

private:
  FrangiTubularnessImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
