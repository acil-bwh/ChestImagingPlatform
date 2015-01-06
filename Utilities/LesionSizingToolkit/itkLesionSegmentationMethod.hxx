/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLesionSegmentationMethod.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLesionSegmentationMethod_hxx
#define __itkLesionSegmentationMethod_hxx

#include "itkLesionSegmentationMethod.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"

// DEBUGGING code:
#include "itkImageFileWriter.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
LesionSegmentationMethod<NDimension>
::LesionSegmentationMethod()
{
  this->SetNumberOfRequiredOutputs( 1 );  // for the Transform

  typedef float                                                 OutputPixelType;
  typedef Image< OutputPixelType, NDimension >                  OutputImageType;
  typedef ImageSpatialObject< NDimension, OutputPixelType >     OutputSpatialObjectType;

  typename OutputSpatialObjectType::Pointer outputObject = OutputSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_ProgressAccumulator = ProgressAccumulator::New();
  this->m_ProgressAccumulator->SetMiniPipelineFilter(this);
}


/**
 * Destructor
 */
template <unsigned int NDimension>
LesionSegmentationMethod<NDimension>
::~LesionSegmentationMethod()
{
}


/**
 * Add a feature generator that will compute the Nth feature to be passed to
 * the segmentation module.
 */
template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::AddFeatureGenerator( FeatureGeneratorType * generator ) 
{
  this->m_FeatureGenerators.push_back( generator );
  this->Modified();
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << "Region of Interest " << this->m_RegionOfInterest.GetPointer() << std::endl;
  os << "Initial Segmentation " << this->m_InitialSegmentation.GetPointer() << std::endl;
  os << "Segmentation Module " << this->m_SegmentationModule.GetPointer() << std::endl;

  os << "Feature generators = ";
  
  FeatureGeneratorConstIterator gitr = this->m_FeatureGenerators.begin();
  FeatureGeneratorConstIterator gend = this->m_FeatureGenerators.end();

  while( gitr != gend )
    {
    os << gitr->GetPointer() << std::endl;
    ++gitr;
    }

}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::GenerateData()
{
  if( !this->m_SegmentationModule )
    {
    itkExceptionMacro("Segmentation Module has not been connected");
    }

  this->UpdateAllFeatureGenerators();

  this->VerifyNumberOfAvailableFeaturesMatchedExpectations();

  this->ConnectFeaturesToSegmentationModule();

  this->ExecuteSegmentationModule();
}


/**
 * Update feature generators
 */
template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::UpdateAllFeatureGenerators()
{
  FeatureGeneratorIterator gitr = this->m_FeatureGenerators.begin();
  FeatureGeneratorIterator gend = this->m_FeatureGenerators.end();

  while( gitr != gend )
    {
    this->m_ProgressAccumulator->RegisterInternalFilter(
            *gitr, 0.5/this->m_FeatureGenerators.size());
    (*gitr)->Update();
    ++gitr;
    }
}


template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::VerifyNumberOfAvailableFeaturesMatchedExpectations() const
{
  const unsigned int expectedNumberOfFeatures = this->m_SegmentationModule->GetExpectedNumberOfFeatures();
  const unsigned int availableNumberOfFeatures = this->m_FeatureGenerators.size();

  if( expectedNumberOfFeatures != availableNumberOfFeatures )
    {
    itkExceptionMacro("Expecting " << expectedNumberOfFeatures << " but only got " << availableNumberOfFeatures );
    }
}


template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::ConnectFeaturesToSegmentationModule()
{
  if( this->m_FeatureGenerators.size() > 0 )
    {
    if( this->m_FeatureGenerators[0]->GetFeature() )
      {
      this->m_SegmentationModule->SetFeature( 
        this->m_FeatureGenerators[0]->GetFeature() );
      }
    }
}

 
template <unsigned int NDimension>
void
LesionSegmentationMethod<NDimension>
::ExecuteSegmentationModule()
{
  this->m_ProgressAccumulator->RegisterInternalFilter(
                      this->m_SegmentationModule, 0.5);
  this->m_SegmentationModule->SetInput( this->m_InitialSegmentation ); 
  this->m_SegmentationModule->Update();
}


} // end namespace itk

#endif
