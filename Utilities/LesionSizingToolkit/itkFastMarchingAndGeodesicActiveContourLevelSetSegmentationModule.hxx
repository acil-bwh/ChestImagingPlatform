/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFastMarchingAndGeodesicActiveContourLevelSetSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFastMarchingAndGeodesicActiveContourLevelSetSegmentationModule_hxx
#define __itkFastMarchingAndGeodesicActiveContourLevelSetSegmentationModule_hxx

#include "itkFastMarchingAndGeodesicActiveContourLevelSetSegmentationModule.h"
#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkProgressAccumulator.h"


namespace itk
{


/**
 * Constructor
 */
template <unsigned int NDimension>
FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule<NDimension>
::FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule()
{
  this->m_FastMarchingModule = FastMarchingModuleType::New();
  this->m_FastMarchingModule->SetDistanceFromSeeds(1.0);
  this->m_FastMarchingModule->SetStoppingValue( 100.0 );
  this->m_FastMarchingModule->InvertOutputIntensitiesOff();
  this->m_GeodesicActiveContourLevelSetModule = GeodesicActiveContourLevelSetModuleType::New();
  this->m_GeodesicActiveContourLevelSetModule->InvertOutputIntensitiesOff();
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule<NDimension>
::~FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
FastMarchingAndGeodesicActiveContourLevelSetSegmentationModule<NDimension>
::GenerateData()
{
  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( this->m_FastMarchingModule, 0.3 );
  progress->RegisterInternalFilter( 
      this->m_GeodesicActiveContourLevelSetModule, 0.7 );

  this->m_FastMarchingModule->SetInput( this->GetInput() );
  this->m_FastMarchingModule->SetFeature( this->GetFeature() );
  this->m_FastMarchingModule->Update();

  m_GeodesicActiveContourLevelSetModule->SetInput( m_FastMarchingModule->GetOutput() );
  m_GeodesicActiveContourLevelSetModule->SetFeature( this->GetFeature() );
  m_GeodesicActiveContourLevelSetModule->SetMaximumRMSError( this->GetMaximumRMSError() );
  m_GeodesicActiveContourLevelSetModule->SetMaximumNumberOfIterations( this->GetMaximumNumberOfIterations() );
  m_GeodesicActiveContourLevelSetModule->SetPropagationScaling( this->GetPropagationScaling() );
  m_GeodesicActiveContourLevelSetModule->SetCurvatureScaling( this->GetCurvatureScaling() );
  m_GeodesicActiveContourLevelSetModule->SetAdvectionScaling( this->GetAdvectionScaling() );
  m_GeodesicActiveContourLevelSetModule->Update();

  this->PackOutputImageInOutputSpatialObject( const_cast< OutputImageType * >(
        dynamic_cast< const OutputSpatialObjectType * >(
        m_GeodesicActiveContourLevelSetModule->GetOutput())->GetImage()) );
}

} // end namespace itk

#endif
