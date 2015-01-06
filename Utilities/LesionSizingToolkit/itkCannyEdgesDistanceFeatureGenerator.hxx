/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkCannyEdgesDistanceFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCannyEdgesDistanceFeatureGenerator_hxx
#define __itkCannyEdgesDistanceFeatureGenerator_hxx

#include "itkCannyEdgesDistanceFeatureGenerator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
CannyEdgesDistanceFeatureGenerator<NDimension>
::CannyEdgesDistanceFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );

  this->m_CastFilter        = CastFilterType::New();
  this->m_DistanceMapFilter = DistanceMapFilterType::New();
  this->m_CannyFilter       = CannyEdgeFilterType::New();

  typename OutputImageSpatialObjectType::Pointer 
    outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Sigma.Fill( 1.0 );
  this->m_UpperThreshold = NumericTraits< InternalPixelType >::max();
  this->m_LowerThreshold = NumericTraits< InternalPixelType >::min();
}


/*
 * Destructor
 */
template <unsigned int NDimension>
CannyEdgesDistanceFeatureGenerator<NDimension>
::~CannyEdgesDistanceFeatureGenerator()
{
}

template <unsigned int NDimension>
void
CannyEdgesDistanceFeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename CannyEdgesDistanceFeatureGenerator<NDimension>::SpatialObjectType *
CannyEdgesDistanceFeatureGenerator<NDimension>
::GetFeature() const
{
  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}


/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
CannyEdgesDistanceFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
CannyEdgesDistanceFeatureGenerator<NDimension>
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

  this->m_CastFilter->SetInput( inputImage );
  this->m_CannyFilter->SetInput( this->m_CastFilter->GetOutput() );
  this->m_DistanceMapFilter->SetInput( this->m_CannyFilter->GetOutput() );

  this->m_CannyFilter->SetSigmaArray( this->m_Sigma );
  this->m_CannyFilter->SetUpperThreshold( this->m_UpperThreshold );
  this->m_CannyFilter->SetLowerThreshold( this->m_LowerThreshold );
  this->m_CannyFilter->SetOutsideValue(NumericTraits<InternalPixelType>::Zero);

  this->m_DistanceMapFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_DistanceMapFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

// Set value of Sigma (isotropic)

template <unsigned int NDimension>
void 
CannyEdgesDistanceFeatureGenerator<NDimension>
::SetSigma( ScalarRealType sigma )
{
  SigmaArrayType sigmas(sigma);
  this->SetSigmaArray(sigmas);
}


// Set value of Sigma (an-isotropic)

template <unsigned int NDimension>
void 
CannyEdgesDistanceFeatureGenerator<NDimension>
::SetSigmaArray( const SigmaArrayType & sigma )
{
  if (this->m_Sigma != sigma)
    {
    this->m_Sigma = sigma;
    this->Modified();
    }
}


// Get the sigma array.
template <unsigned int NDimension>
typename CannyEdgesDistanceFeatureGenerator<NDimension>::SigmaArrayType
CannyEdgesDistanceFeatureGenerator<NDimension>
::GetSigmaArray() const
{
  return m_Sigma;
}


// Get the sigma scalar. If the sigma is anisotropic, we will just
// return the sigma along the first dimension.
template <unsigned int NDimension>
typename CannyEdgesDistanceFeatureGenerator<NDimension>::ScalarRealType
CannyEdgesDistanceFeatureGenerator<NDimension>
::GetSigma() const
{
  return m_Sigma[0];
}


} // end namespace itk

#endif
