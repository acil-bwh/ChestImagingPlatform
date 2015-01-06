/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkCannyEdgesDistanceAdvectionFieldFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCannyEdgesDistanceAdvectionFieldFeatureGenerator_hxx
#define __itkCannyEdgesDistanceAdvectionFieldFeatureGenerator_hxx

#include "itkCannyEdgesDistanceAdvectionFieldFeatureGenerator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
::CannyEdgesDistanceAdvectionFieldFeatureGenerator()
{
  this->SetNumberOfRequiredInputs( 1 );

  this->m_CastFilter        = CastFilterType::New();
  this->m_DistanceMapFilter = DistanceMapFilterType::New();
  this->m_CannyFilter       = CannyEdgeFilterType::New();
  this->m_MultiplyFilter    = MultiplyFilterType::New();
  this->m_GradientFilter    = GradientFilterType::New();

  typename OutputImageSpatialObjectType::Pointer 
    outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_Sigma =  1.0;
  this->m_UpperThreshold = NumericTraits< InternalPixelType >::max();
  this->m_LowerThreshold = NumericTraits< InternalPixelType >::min();
}


/*
 * Destructor
 */
template <unsigned int NDimension>
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
::~CannyEdgesDistanceAdvectionFieldFeatureGenerator()
{
}

template <unsigned int NDimension>
void
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>::SpatialObjectType *
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
::GetFeature() const
{
  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}


/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
CannyEdgesDistanceAdvectionFieldFeatureGenerator<NDimension>
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

  this->m_CannyFilter->SetSigma( this->m_Sigma );
  this->m_CannyFilter->SetUpperThreshold( this->m_UpperThreshold );
  this->m_CannyFilter->SetLowerThreshold( this->m_LowerThreshold );
  this->m_CannyFilter->SetOutsideValue(NumericTraits<InternalPixelType>::Zero);

  this->m_DistanceMapFilter->Update();

  m_GradientFilter->SetInput(m_DistanceMapFilter->GetOutput());
  m_GradientFilter->Update();

  m_MultiplyFilter->SetInput1(m_GradientFilter->GetOutput());
  m_MultiplyFilter->SetInput2(m_DistanceMapFilter->GetOutput());
  m_MultiplyFilter->Update();  
  
  typename OutputImageType::Pointer outputImage = this->m_MultiplyFilter->GetOutput();

  outputImage->DisconnectPipeline();

  OutputImageSpatialObjectType * outputObject =
    dynamic_cast< OutputImageSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
