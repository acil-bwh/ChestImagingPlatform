/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSatoVesselnessFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSatoVesselnessFeatureGenerator_hxx
#define __itkSatoVesselnessFeatureGenerator_hxx

#include "itkSatoVesselnessFeatureGenerator.h"
#include "itkProgressAccumulator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
SatoVesselnessFeatureGenerator<NDimension>
::SatoVesselnessFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );

  this->m_HessianFilter = HessianFilterType::New();
  this->m_VesselnessFilter = VesselnessMeasureFilterType::New();

  this->m_HessianFilter->ReleaseDataFlagOn();
  this->m_VesselnessFilter->ReleaseDataFlagOn();

  typename OutputImageSpatialObjectType::Pointer outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Sigma =  1.0;
  this->m_Alpha1 = 0.5;
  this->m_Alpha2 = 2.0;

  this->m_VesselEnhancingDiffusionFilter = VesselEnhancingDiffusionFilterType::New();
  this->m_UseVesselEnhancingDiffusion = false;
}


/*
 * Destructor
 */
template <unsigned int NDimension>
SatoVesselnessFeatureGenerator<NDimension>
::~SatoVesselnessFeatureGenerator()
{
}

template <unsigned int NDimension>
void
SatoVesselnessFeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename SatoVesselnessFeatureGenerator<NDimension>::SpatialObjectType *
SatoVesselnessFeatureGenerator<NDimension>
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
SatoVesselnessFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Vesselness Sigma " << this->m_Sigma << std::endl;
  os << indent << "Vesselness Alpha1 " << this->m_Alpha1 << std::endl;
  os << indent << "Vesselness Alpha2 " << this->m_Alpha2 << std::endl;
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
SatoVesselnessFeatureGenerator<NDimension>
::GenerateData()
{
  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);

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


  // Two alternative routes :
  //
  //   Input -> VED -> Sato
  //   Input -> Hessian -> Sato
  //
  if (this->m_UseVesselEnhancingDiffusion)
    {
    // Set the default scales for the vessel enhancing diffusion filter.

    typename InputImageType::SpacingType spacing = inputImage->GetSpacing();
    double minSpacing = itk::NumericTraits< double >::max();
    for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
      {
      if (minSpacing > spacing[i])
        {
        minSpacing = spacing[i];
        }
      }

    std::vector< typename VesselEnhancingDiffusionFilterType::Precision > scales(5);
    scales[0] = 1.0    * minSpacing;
    scales[1] = 1.6067 * minSpacing;
    scales[2] = 2.5833 * minSpacing;
    scales[3] = 4.15   * minSpacing;
    scales[4] = 6.66   * minSpacing;
    this->m_VesselEnhancingDiffusionFilter->SetDefaultPars();
    this->m_VesselEnhancingDiffusionFilter->SetScales(scales);

    this->m_VesselEnhancingDiffusionFilter->SetInput( inputImage );
    this->m_HessianFilter->SetInput( m_VesselEnhancingDiffusionFilter->GetOutput() );
    this->m_VesselnessFilter->SetInput( this->m_HessianFilter->GetOutput() );

    progress->RegisterInternalFilter( this->m_VesselEnhancingDiffusionFilter, .8 );
    progress->RegisterInternalFilter( this->m_HessianFilter, .1 );
    progress->RegisterInternalFilter( this->m_VesselnessFilter, .1 );
    }
  else
    {
    this->m_HessianFilter->SetInput( inputImage );
    this->m_VesselnessFilter->SetInput( this->m_HessianFilter->GetOutput() );
    progress->RegisterInternalFilter( this->m_HessianFilter, .7 );
    progress->RegisterInternalFilter( this->m_VesselnessFilter, .3 );
    }

  this->m_HessianFilter->SetSigma( this->m_Sigma );
  this->m_VesselnessFilter->SetAlpha1( this->m_Alpha1 );
  this->m_VesselnessFilter->SetAlpha2( this->m_Alpha2 );

  this->m_VesselnessFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_VesselnessFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
