/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkBinaryThresholdFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkBinaryThresholdFeatureGenerator_hxx
#define __itkBinaryThresholdFeatureGenerator_hxx

#include "itkBinaryThresholdFeatureGenerator.h"
#include "itkProgressAccumulator.h"

namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
BinaryThresholdFeatureGenerator<NDimension>
::BinaryThresholdFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );

  this->m_BinaryThresholdFilter = BinaryThresholdFilterType::New();

  this->m_BinaryThresholdFilter->ReleaseDataFlagOn();

  typename OutputImageSpatialObjectType::Pointer outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Threshold = 128.0;
}


/**
 * Destructor
 */
template <unsigned int NDimension>
BinaryThresholdFeatureGenerator<NDimension>
::~BinaryThresholdFeatureGenerator()
{
}

/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
BinaryThresholdFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
BinaryThresholdFeatureGenerator<NDimension>
::GenerateData()
{
  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( this->m_BinaryThresholdFilter, 1.0 );  

  typename InputImageSpatialObjectType::ConstPointer inputObject =
    dynamic_cast<const InputImageSpatialObjectType * >( this->ProcessObject::GetInput(0) );

  if( !inputObject )
    {
    itkExceptionMacro("Missing input spatial object or incorrect type");
    }

  const InputImageType * inputImage = inputObject->GetImage();

  if( !inputImage )
    {
    itkExceptionMacro("Missing input image");
    }

  this->m_BinaryThresholdFilter->SetInput( inputImage );
  this->m_BinaryThresholdFilter->SetLowerThreshold( this->m_Threshold );
  this->m_BinaryThresholdFilter->SetUpperThreshold( itk::NumericTraits< OutputPixelType >::max() );
  this->m_BinaryThresholdFilter->SetOutsideValue( 0.0 );
  this->m_BinaryThresholdFilter->SetInsideValue( 1.0 );

  this->m_BinaryThresholdFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_BinaryThresholdFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
