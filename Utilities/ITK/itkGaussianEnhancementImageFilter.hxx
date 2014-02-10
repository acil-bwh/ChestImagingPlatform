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
#ifndef __itkGaussianEnhancementImageFilter_hxx
#define __itkGaussianEnhancementImageFilter_hxx

#include "itkGaussianEnhancementImageFilter.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template < typename TInPixel, typename TOutPixel >
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::GaussianEnhancementImageFilter()
{
  this->m_UnaryFunctor = NULL;
  this->m_BinaryFunctor = NULL;
  this->m_UnaryFunctorFilter = UnaryFunctorImageFilterType::New();//needed to be global?
  this->m_BinaryFunctorFilter = BinaryFunctorImageFilterType::New();
  this->m_Sigma = 1.0;
  this->m_Rescale = true;
  this->m_NormalizeAcrossScale = true;

  // Construct the gradient magnitude filter
  this->m_GradientMagnitudeFilter = GradientMagnitudeFilterType::New();
  this->m_GradientMagnitudeFilter->SetNormalizeAcrossScale( this->m_NormalizeAcrossScale );

  // Construct the Hessian filter
  this->m_HessianFilter = HessianFilterType::New();
  this->m_HessianFilter->SetNormalizeAcrossScale( this->m_NormalizeAcrossScale );

  // Construct the eigenvalue filter
  this->m_SymmetricEigenValueFilter = EigenAnalysisFilterType::New();
  this->m_SymmetricEigenValueFilter->SetDimension( ImageDimension );
  this->m_SymmetricEigenValueFilter->OrderEigenValuesBy(
    EigenAnalysisFilterType::FunctorType::OrderByValue );//OrderByMagnitude?

  // Construct the rescale filter
  this->m_RescaleFilter = RescaleFilterType::New();
  this->m_RescaleFilter->SetOutputMinimum( 0.0 );
  this->m_RescaleFilter->SetOutputMaximum( 1.0 );

  // Allow progressive memory release
  this->m_HessianFilter->ReleaseDataFlagOn();
  this->m_GradientMagnitudeFilter->ReleaseDataFlagOn();
  this->m_SymmetricEigenValueFilter->ReleaseDataFlagOn();
  this->m_RescaleFilter->ReleaseDataFlagOn();

} // end Constructor


/**
 * ********************* SetUnaryFunctor ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::SetUnaryFunctor( UnaryFunctorBaseType * _arg )
{
  if ( this->m_UnaryFunctor != _arg )
  {
    // Only one of them should be initialized
    this->m_UnaryFunctor = _arg;
    this->m_UnaryFunctorFilter->SetFunctor( _arg );
    this->m_BinaryFunctor = NULL;
    this->Modified();
  }
} // end SetUnaryFunctor()


/**
 * ********************* SetBinaryFunctor ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::SetBinaryFunctor( BinaryFunctorBaseType * _arg )
{
  if ( this->m_BinaryFunctor != _arg )
  {
    // Only one of them should be initialized
    this->m_BinaryFunctor = _arg;
    this->m_BinaryFunctorFilter->SetFunctor( _arg );
    this->m_UnaryFunctor = NULL;
    this->Modified();
  }
} // end SetBinaryFunctor()


/**
 * ********************* SetNumberOfThreads ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::SetNumberOfThreads( ThreadIdType nt )
{
  Superclass::SetNumberOfThreads( nt );

  this->m_GradientMagnitudeFilter->SetNumberOfThreads( nt );
  this->m_HessianFilter->SetNumberOfThreads( nt );
  this->m_SymmetricEigenValueFilter->SetNumberOfThreads( nt );
  this->m_RescaleFilter->SetNumberOfThreads( nt );

  if ( this->m_UnaryFunctorFilter.IsNotNull() )
  {
    this->m_UnaryFunctorFilter->SetNumberOfThreads( nt );
  }
  else if( this->m_BinaryFunctorFilter.IsNotNull() )
  {
    this->m_BinaryFunctorFilter->SetNumberOfThreads( nt );
  }

  if ( this->GetNumberOfThreads() != ( nt < 1 ? 1 : ( nt > ITK_MAX_THREADS ? ITK_MAX_THREADS : nt ) ) )
  {
    this->Modified();
  }
} // end SetNumberOfThreads()


/**
 * ********************* SetNormalizeAcrossScale ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::SetNormalizeAcrossScale( bool normalize )
{
  itkDebugMacro( "Setting NormalizeAcrossScale to " << normalize );
  if( this->m_NormalizeAcrossScale != normalize )
  {
    this->m_NormalizeAcrossScale = normalize;

    this->m_GradientMagnitudeFilter->SetNormalizeAcrossScale( this->m_NormalizeAcrossScale );
    this->m_HessianFilter->SetNormalizeAcrossScale( this->m_NormalizeAcrossScale );

    this->Modified();
  }
} // end SetNormalizeAcrossScale()


/**
 * ********************* GenerateData ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::GenerateData( void )
{
  if ( this->m_UnaryFunctor.IsNull()
    && this->m_BinaryFunctor.IsNull() )
  {
    itkExceptionMacro( << "ERROR: Missing Functor. "
      << "Please provide functor for multi scale framework." );
  }

  // Define if we going to use gradient magnitude based on if BinaryFunctorFilter
  // has been provided
  if ( this->m_BinaryFunctor.IsNotNull() )
  {
    // Calculate the gradient magnitude scalar image.
    this->m_GradientMagnitudeFilter->SetInput( this->GetInput() );
    this->m_GradientMagnitudeFilter->SetSigma( this->m_Sigma );
    this->m_GradientMagnitudeFilter->Update();
  }

  // Calculate the eigenvalue vector image.
  this->m_HessianFilter->SetInput( this->GetInput() );
  this->m_HessianFilter->SetSigma( this->m_Sigma );

  this->m_SymmetricEigenValueFilter->SetInput( this->m_HessianFilter->GetOutput() );
  this->m_SymmetricEigenValueFilter->Update();

  if ( this->m_BinaryFunctor.IsNotNull() )
  {
    // Calculate binary functor filter.
    this->m_BinaryFunctorFilter->SetInput1(
      this->m_GradientMagnitudeFilter->GetOutput() );
    this->m_BinaryFunctorFilter->SetInput2(
      this->m_SymmetricEigenValueFilter->GetOutput() );
    this->m_BinaryFunctorFilter->Update();
  }
  else
  {
    // Calculate unary functor filter.
    this->m_UnaryFunctorFilter->SetInput(
      this->m_SymmetricEigenValueFilter->GetOutput() );
    this->m_UnaryFunctorFilter->Update();
  }

  // Apply rescale
  if( this->m_Rescale )
  {
    // Rescale the output to [0,1].
    if ( this->m_BinaryFunctor.IsNotNull() )
    {
      this->m_RescaleFilter->SetInput( this->m_BinaryFunctorFilter->GetOutput() );
    }
    else
    {
      this->m_RescaleFilter->SetInput( this->m_UnaryFunctorFilter->GetOutput() );
    }
    this->m_RescaleFilter->Update();

    // Put the output of the rescale filter to this filter's output.
    this->GraftOutput( this->m_RescaleFilter->GetOutput() );
  }
  else
  {
    if ( this->m_BinaryFunctor.IsNotNull() )
    {
      this->GraftOutput( this->m_BinaryFunctorFilter->GetOutput() );
    }
    else
    {
      this->GraftOutput( this->m_UnaryFunctorFilter->GetOutput() );
    }
  }
} // end GenerateData()


/**
 * ********************* PrintSelf ****************************
 */

template < typename TInPixel, typename TOutPixel >
void
GaussianEnhancementImageFilter< TInPixel, TOutPixel >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Sigma: " << this->m_Sigma << std::endl;
  os << indent << "Rescale: " << this->m_Rescale << std::endl;
  os << indent << "NormalizeAcrossScale: " << this->m_NormalizeAcrossScale << std::endl;

  Indent nextIndent = indent.GetNextIndent();
  if ( this->m_BinaryFunctorFilter.IsNotNull() )
  {
    this->m_BinaryFunctorFilter->Print( os, nextIndent );
  }
  else
  {
    this->m_UnaryFunctorFilter->Print( os, nextIndent );
  }
} // end PrintSelf()


} // end namespace itk

#endif
