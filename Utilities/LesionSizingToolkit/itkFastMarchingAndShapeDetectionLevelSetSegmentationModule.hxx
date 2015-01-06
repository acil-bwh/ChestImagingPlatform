/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFastMarchingAndShapeDetectionLevelSetSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFastMarchingAndShapeDetectionLevelSetSegmentationModule_hxx
#define __itkFastMarchingAndShapeDetectionLevelSetSegmentationModule_hxx

#include "itkFastMarchingAndShapeDetectionLevelSetSegmentationModule.h"
#include "itkShapeDetectionLevelSetImageFilter.h"
#include "itkProgressAccumulator.h"


namespace itk
{


/**
 * Constructor
 */
template <unsigned int NDimension>
FastMarchingAndShapeDetectionLevelSetSegmentationModule<NDimension>
::FastMarchingAndShapeDetectionLevelSetSegmentationModule()
{
  this->m_FastMarchingModule = FastMarchingModuleType::New();
  this->m_FastMarchingModule->SetDistanceFromSeeds(1.0);
  this->m_FastMarchingModule->SetStoppingValue(100.0);
  this->m_FastMarchingModule->InvertOutputIntensitiesOff();
  this->m_ShapeDetectionLevelSetModule = ShapeDetectionLevelSetModuleType::New();
  this->m_ShapeDetectionLevelSetModule->InvertOutputIntensitiesOff();
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FastMarchingAndShapeDetectionLevelSetSegmentationModule<NDimension>
::~FastMarchingAndShapeDetectionLevelSetSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
FastMarchingAndShapeDetectionLevelSetSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
FastMarchingAndShapeDetectionLevelSetSegmentationModule<NDimension>
::GenerateData()
{
  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( this->m_FastMarchingModule, 0.3 );
  progress->RegisterInternalFilter( 
      this->m_ShapeDetectionLevelSetModule, 0.7 );
  
  this->m_FastMarchingModule->SetInput( this->GetInput() );
  this->m_FastMarchingModule->SetFeature( this->GetFeature() );
  this->m_FastMarchingModule->Update();

  m_ShapeDetectionLevelSetModule->SetInput( m_FastMarchingModule->GetOutput() );
  m_ShapeDetectionLevelSetModule->SetFeature( this->GetFeature() );
  m_ShapeDetectionLevelSetModule->SetMaximumRMSError( this->GetMaximumRMSError() );
  m_ShapeDetectionLevelSetModule->SetMaximumNumberOfIterations( this->GetMaximumNumberOfIterations() );
  m_ShapeDetectionLevelSetModule->SetPropagationScaling( this->GetPropagationScaling() );
  m_ShapeDetectionLevelSetModule->SetCurvatureScaling( this->GetCurvatureScaling() );
  m_ShapeDetectionLevelSetModule->Update();

  this->PackOutputImageInOutputSpatialObject( const_cast< OutputImageType * >(
        dynamic_cast< const OutputSpatialObjectType * >(
        m_ShapeDetectionLevelSetModule->GetOutput())->GetImage()) );
}

} // end namespace itk

#endif
