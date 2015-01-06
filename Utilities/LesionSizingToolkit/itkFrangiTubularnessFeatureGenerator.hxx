/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFrangiTubularnessFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFrangiTubularnessFeatureGenerator_hxx
#define __itkFrangiTubularnessFeatureGenerator_hxx

#include "itkFrangiTubularnessFeatureGenerator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
FrangiTubularnessFeatureGenerator<NDimension>
::FrangiTubularnessFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );

  this->m_HessianFilter = HessianFilterType::New();
  this->m_EigenAnalysisFilter = EigenAnalysisFilterType::New();
  this->m_SheetnessFilter = SheetnessFilterType::New();

  this->m_HessianFilter->ReleaseDataFlagOn();
  this->m_EigenAnalysisFilter->ReleaseDataFlagOn();
  this->m_SheetnessFilter->ReleaseDataFlagOn();

  typename OutputImageSpatialObjectType::Pointer outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Sigma =  1.0;
  this->m_SheetnessNormalization = 0.5;
  this->m_BloobinessNormalization = 2.0;
  this->m_NoiseNormalization = 1.0;
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FrangiTubularnessFeatureGenerator<NDimension>
::~FrangiTubularnessFeatureGenerator()
{
}

template <unsigned int NDimension>
void
FrangiTubularnessFeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename FrangiTubularnessFeatureGenerator<NDimension>::SpatialObjectType *
FrangiTubularnessFeatureGenerator<NDimension>
::GetFeature() const
{
  if (this->GetNumberOfOutputs() < 1)
    {
    return 0;
    }

  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));

}


/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
FrangiTubularnessFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
FrangiTubularnessFeatureGenerator<NDimension>
::GenerateData()
{
  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( this->m_HessianFilter, .5 );
  progress->RegisterInternalFilter( this->m_EigenAnalysisFilter, .25 );
  progress->RegisterInternalFilter( this->m_SheetnessFilter, .25 );

  typename InputImageSpatialObjectType::ConstPointer inputObject = 
    dynamic_cast<const InputImageSpatialObjectType * >( this->ProcessObject::GetInput(0) );

  if( !inputObject )
    {
    itkExceptionMacro("Missing input spatial object");
    }

  const InputImageType * inputImage = inputObject->GetImage();

  if( !inputImage )
    {
    itkExceptionMacro("Missing input image");
    }

  this->m_HessianFilter->SetInput( inputImage );
  this->m_EigenAnalysisFilter->SetInput( this->m_HessianFilter->GetOutput() );
  this->m_SheetnessFilter->SetInput( this->m_EigenAnalysisFilter->GetOutput() );

  this->m_HessianFilter->SetSigma( this->m_Sigma );
  this->m_EigenAnalysisFilter->SetDimension( Dimension );
  this->m_SheetnessFilter->SetSheetnessNormalization( this->m_SheetnessNormalization );
  this->m_SheetnessFilter->SetBloobinessNormalization( this->m_BloobinessNormalization );
  this->m_SheetnessFilter->SetNoiseNormalization( this->m_NoiseNormalization );

  this->m_SheetnessFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_SheetnessFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
