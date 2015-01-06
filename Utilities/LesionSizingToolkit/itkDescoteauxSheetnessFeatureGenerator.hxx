/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkDescoteauxSheetnessFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDescoteauxSheetnessFeatureGenerator_hxx
#define __itkDescoteauxSheetnessFeatureGenerator_hxx

#include "itkDescoteauxSheetnessFeatureGenerator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
DescoteauxSheetnessFeatureGenerator<NDimension>
::DescoteauxSheetnessFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );

  this->m_HessianFilter = HessianFilterType::New();
  this->m_EigenAnalysisFilter = EigenAnalysisFilterType::New();
  this->m_SheetnessFilter = SheetnessFilterType::New();
  this->m_RescaleFilter = RescaleFilterType::New();

  // Allow progressive memory release
  this->m_HessianFilter->ReleaseDataFlagOn();
  this->m_EigenAnalysisFilter->ReleaseDataFlagOn();
  this->m_SheetnessFilter->ReleaseDataFlagOn();
  this->m_RescaleFilter->ReleaseDataFlagOn();

  typename OutputImageSpatialObjectType::Pointer outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Sigma =  1.0;
  this->m_SheetnessNormalization = 0.5;
  this->m_BloobinessNormalization = 2.0;
  this->m_NoiseNormalization = 1.0;
  this->m_DetectBrightSheets = true;
}


/**
 * Destructor
 */
template <unsigned int NDimension>
DescoteauxSheetnessFeatureGenerator<NDimension>
::~DescoteauxSheetnessFeatureGenerator()
{
}

template <unsigned int NDimension>
void
DescoteauxSheetnessFeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename DescoteauxSheetnessFeatureGenerator<NDimension>::SpatialObjectType *
DescoteauxSheetnessFeatureGenerator<NDimension>
::GetFeature() const
{
  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}


/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
DescoteauxSheetnessFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
DescoteauxSheetnessFeatureGenerator<NDimension>
::GenerateData()
{
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

  this->m_HessianFilter->SetInput( inputImage );
  this->m_EigenAnalysisFilter->SetInput( this->m_HessianFilter->GetOutput() );
  this->m_SheetnessFilter->SetInput( this->m_EigenAnalysisFilter->GetOutput() );
  this->m_RescaleFilter->SetInput( this->m_SheetnessFilter->GetOutput() );

  this->m_HessianFilter->SetSigma( this->m_Sigma );
  this->m_EigenAnalysisFilter->SetDimension( Dimension );
  this->m_SheetnessFilter->SetSheetnessNormalization( this->m_SheetnessNormalization );
  this->m_SheetnessFilter->SetBloobinessNormalization( this->m_BloobinessNormalization );
  this->m_SheetnessFilter->SetNoiseNormalization( this->m_NoiseNormalization );
  this->m_SheetnessFilter->SetDetectBrightSheets( this->m_DetectBrightSheets );

  this->m_RescaleFilter->SetOutputMinimum( 0.0 );
  this->m_RescaleFilter->SetOutputMaximum( 1.0 );

  this->m_RescaleFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_RescaleFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
