/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkIsotropicResamplerImageFilter.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkIsotropicResamplerImageFilter_hxx
#define __itkIsotropicResamplerImageFilter_hxx

#include "itkIsotropicResamplerImageFilter.h"
#include "itkProgressAccumulator.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
IsotropicResamplerImageFilter<TInputImage, TOutputImage>::
IsotropicResamplerImageFilter()
{
  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );

  this->m_OutputSpacing.Fill( 0.2 );  // 0.2 mm
  this->m_DefaultPixelValue = static_cast< OutputImagePixelType >(0.0);
  this->m_ResampleFilter = ResampleFilterType::New();
}
  
template <class TInputImage, class TOutputImage>
IsotropicResamplerImageFilter<TInputImage, TOutputImage>::
~IsotropicResamplerImageFilter()
{
}
 
template< class TInputImage, class TOutputImage >
void
IsotropicResamplerImageFilter< TInputImage, TOutputImage >
::GenerateOutputInformation()
{
  // call the superclass' implementation of this method
  this->Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
    return;
    }

  const InputImageType * inputImage = this->GetInput();
  if( !inputImage )
    {
    itkExceptionMacro("Missing input image");
    }
  
  const SpacingType & inputSpacing = inputImage->GetSpacing();

  SizeType inputSize = inputImage->GetLargestPossibleRegion().GetSize(), finalSize;
  for (unsigned int i = 0; i < ImageDimension; i++)
    {
    const double dx = inputSize[i] * inputSpacing[i] / m_OutputSpacing[i];
    finalSize[i] = static_cast<SizeValueType>( dx );
    }
  
  typename TOutputImage::RegionType outputLargestPossibleRegion;
  typename TOutputImage::RegionType::IndexType index;
  index.Fill(0);
  outputLargestPossibleRegion.SetSize( finalSize );
  outputLargestPossibleRegion.SetIndex( index );
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
  outputPtr->SetSpacing( m_OutputSpacing );
  outputPtr->SetOrigin( inputImage->GetOrigin() );

#if ITK_VERSION_MAJOR > 3 || (ITK_VERSION_MAJOR == 3 && ITK_VERSION_MINOR >= 10)
  outputPtr->SetDirection( inputImage->GetDirection() );
#endif
}

template< class TInputImage, class TOutputImage >
void
IsotropicResamplerImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  // Report progress.
  typename ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);

  const InputImageType * inputImage = this->GetInput();
  if( !inputImage )
    {
    itkExceptionMacro("Missing input image");
    }

  if (m_OutputSpacing == inputImage->GetSpacing())
    {
    // No need to resample. Desiered output spacing is the same as the input 
    // spacing. Let's just graft the output and be done with it.
    this->GraftOutput( const_cast< InputImageType * >(this->GetInput()) );
    return;
    }

  typedef itk::IdentityTransform< double, ImageDimension >  TransformType;

  typename TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  typedef itk::BSplineInterpolateImageFunction< InputImageType, double >  BSplineInterpolatorType;

  typename BSplineInterpolatorType::Pointer bsplineInterpolator = BSplineInterpolatorType::New();

#if ITK_VERSION_MAJOR > 3 || (ITK_VERSION_MAJOR == 3 && ITK_VERSION_MINOR >= 10)
  bsplineInterpolator->UseImageDirectionOn();
#endif
  bsplineInterpolator->SetSplineOrder( 3 );

  const SpacingType & inputSpacing = inputImage->GetSpacing();
  SizeType inputSize = inputImage->GetLargestPossibleRegion().GetSize(), finalSize;
  for (unsigned int i = 0; i < ImageDimension; i++)
    {
    const double dx = inputSize[i] * inputSpacing[i] / m_OutputSpacing[i];
    finalSize[i] = static_cast<SizeValueType>( dx );
    }

  this->m_ResampleFilter->SetTransform( transform );
  this->m_ResampleFilter->SetInterpolator( bsplineInterpolator );
  this->m_ResampleFilter->SetDefaultPixelValue( this->m_DefaultPixelValue );
  this->m_ResampleFilter->SetOutputSpacing( m_OutputSpacing );
  this->m_ResampleFilter->SetOutputOrigin( inputImage->GetOrigin() );
  this->m_ResampleFilter->SetOutputDirection( inputImage->GetDirection() );
  this->m_ResampleFilter->SetSize( finalSize );
  this->m_ResampleFilter->SetInput( inputImage );

  progress->RegisterInternalFilter( this->m_ResampleFilter, 1.0 );  

  this->m_ResampleFilter->Update();

  this->GraftOutput( this->m_ResampleFilter->GetOutput() );
}

template <class TInputImage, class TOutputImage>
void IsotropicResamplerImageFilter< TInputImage,TOutputImage >
::SetAbortGenerateData( const bool abort )
{
  this->Superclass::SetAbortGenerateData(abort);
  this->m_ResampleFilter->SetAbortGenerateData(abort);
}


template <class TInputImage, class TOutputImage>
void 
IsotropicResamplerImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

}//end of itk namespace

#endif
