/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSatoVesselnessSigmoidFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSatoVesselnessSigmoidFeatureGenerator_hxx
#define __itkSatoVesselnessSigmoidFeatureGenerator_hxx

#include "itkSatoVesselnessSigmoidFeatureGenerator.h"
#include "itkProgressAccumulator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
SatoVesselnessSigmoidFeatureGenerator<NDimension>
::SatoVesselnessSigmoidFeatureGenerator()
{
  this->m_SigmoidFilter = SigmoidFilterType::New();

  this->m_SigmoidFilter->ReleaseDataFlagOn();

  this->m_SigmoidAlpha =  -1.0;
  this->m_SigmoidBeta = 90.0;
}


/*
 * Destructor
 */
template <unsigned int NDimension>
SatoVesselnessSigmoidFeatureGenerator<NDimension>
::~SatoVesselnessSigmoidFeatureGenerator()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
SatoVesselnessSigmoidFeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Sigmoid Alpha " << this->m_SigmoidAlpha << std::endl;
  os << indent << "Sigmoid Beta " << this->m_SigmoidBeta << std::endl;
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
SatoVesselnessSigmoidFeatureGenerator<NDimension>
::GenerateData()
{
  this->Superclass::GenerateData();

  // Report progress. Actually, the superclass will report upto 1 in
  // the superclass's generate data method. This will start again
  // from 0, but that's ok. :)
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( this->m_SigmoidFilter, 1.0 ); 

  //
  // Take the output of the superclass, and do further processing on it.
  // 
  typename OutputImageSpatialObjectType::Pointer outputObject =
    dynamic_cast<OutputImageSpatialObjectType * >( this->GetInternalFeature() );

  const OutputImageType * inputImage = outputObject->GetImage();

  this->m_SigmoidFilter->SetInput( inputImage );

  this->m_SigmoidFilter->SetAlpha( this->m_SigmoidAlpha );
  this->m_SigmoidFilter->SetBeta( this->m_SigmoidBeta );

  this->m_SigmoidFilter->SetOutputMinimum( 0.0 );
  this->m_SigmoidFilter->SetOutputMaximum( 1.0 );

  this->m_SigmoidFilter->Update();

  typename OutputImageType::Pointer outputImage = this->m_SigmoidFilter->GetOutput();

  outputImage->DisconnectPipeline();

  outputObject->SetImage( outputImage );
}

} // end namespace itk

#endif
