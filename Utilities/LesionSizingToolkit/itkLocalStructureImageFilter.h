/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLocalStructureImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLocalStructureImageFilter_h
#define __itkLocalStructureImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{

/** \class LocalStructureImageFilter
 *
 * \brief Computes local similarity to geometrical structures using second
 * derivative operations.
 *
 * Based on the paper
 *
 *      "Tissue Classification Based on 3D Local Intensity Structures for
 *      Volume Rendering".
 *
 *  by
 *
 *      Y. Sato, C-F. Westin, A. Bhalerao, S. Nakajima,
 *      N. Shiraga, S. Tamura, R. Kikinis.
 *
 * IEEE Transactions on Visualization and Computer Graphics
 * Vol 6. No. 2. April-June 2000.
 *
 * \ingroup IntensityImageFilters  Multithreaded
 * \ingroup LesionSizingToolkit
 */
namespace Function {

template< class TInput, class TOutput>
class LocalStructure
{
public:
  LocalStructure()
    {
    m_Alpha = 0.25; // suggested value in the paper
    m_Gamma = 0.50; // suggested value in the paper;
    }
  ~LocalStructure() {}
  bool operator!=( const LocalStructure & ) const
    {
    return false;
    }
  bool operator==( const LocalStructure & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A )
    {
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

    //
    // Avoid divisions by zero.
    //
    if( l3 < vnl_math::eps )
      {
      return 0.0;
      }

    const double L3 = vnl_math_abs( static_cast<double>( a3 ) );
    const double W = WeightFunctionOmega( a2, a3 );
    const double F = WeightFunctionOmega( a1, a3 );

    const double sheetness  = L3 * W * F;

    return static_cast<TOutput>( sheetness );
    }
  inline double WeightFunctionOmega( double ls, double lt ) const
    {
    if( ls <= 0.0 && lt <= ls )
      {
      return ( 1 + std::pow( ls / vnl_math_abs( lt ), m_Gamma ) );
      }
    const double abslt = vnl_math_abs( lt );
    if( ls > 0.0  &&  abslt / m_Gamma > ls )
      {
      return std::pow( 1 - m_Alpha * ls / vnl_math_abs( lt ), m_Gamma );
      }
    return 0.0;
    }
   inline double WeightFunctionPhi( double ls, double lt ) const
    {
    if( ls < 0.0 && lt <= ls )
      {
      return std::pow( ( ls / lt ), m_Gamma );
      }
    return 0.0;
    }

  void SetAlpha( double value )
    {
    this->m_Alpha = value;
    }

  void SetGamma( double value )
    {
    this->m_Gamma = value;
    }

private:
  double m_Alpha;
  double m_Gamma;
};
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT LocalStructureImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage,
                        Function::LocalStructure< typename TInputImage::PixelType,
                                       typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef LocalStructureImageFilter           Self;
  typedef UnaryFunctorImageFilter<
    TInputImage,TOutputImage,
    Function::LocalStructure<
      typename TInputImage::PixelType,
      typename TOutputImage::PixelType> >     Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LocalStructureImageFilter,
               UnaryFunctorImageFilter);

  /** Set the normalization term for sheetness */
  void SetAlpha( double value )
    {
    this->GetFunctor().SetAlpha( value );
    }

  /** Set the normalization term for bloobiness. */
  void SetGamma( double value )
    {
    this->GetFunctor().SetGamma( value );
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
  LocalStructureImageFilter() {}
  virtual ~LocalStructureImageFilter() {}

private:
  LocalStructureImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
