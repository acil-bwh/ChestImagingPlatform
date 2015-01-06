/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFeatureAggregator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFeatureAggregator_hxx
#define __itkFeatureAggregator_hxx

#include "itkFeatureAggregator.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
FeatureAggregator<NDimension>
::FeatureAggregator()
{
  this->SetNumberOfRequiredOutputs( 1 );

  typename OutputImageSpatialObjectType::Pointer outputObject = OutputImageSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );

  this->m_ProgressAccumulator = ProgressAccumulator::New();
  this->m_ProgressAccumulator->SetMiniPipelineFilter(this);
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FeatureAggregator<NDimension>
::~FeatureAggregator()
{
}


/**
 * Add a feature generator that will compute the Nth feature to be passed to
 * the segmentation module.
 */
template <unsigned int NDimension>
void
FeatureAggregator<NDimension>
::AddFeatureGenerator( FeatureGeneratorType * generator )
{
  this->m_FeatureGenerators.push_back( generator );
}


template <unsigned int NDimension>
unsigned int
FeatureAggregator<NDimension>
::GetNumberOfInputFeatures() const
{
  return this->m_FeatureGenerators.size();
}


template <unsigned int NDimension>
const typename FeatureAggregator<NDimension>::InputFeatureType *
FeatureAggregator<NDimension>
::GetInputFeature( unsigned int featureId ) const
{
  if( featureId >= this->GetNumberOfInputFeatures() )
    {
    itkExceptionMacro("Feature Id" << featureId << " doesn't exist");
    }
  return this->m_FeatureGenerators[featureId]->GetFeature();
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
FeatureAggregator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Feature generators = ";

  FeatureGeneratorConstIterator gitr = this->m_FeatureGenerators.begin();
  FeatureGeneratorConstIterator gend = this->m_FeatureGenerators.end();

  while( gitr != gend )
    {
    os << indent << gitr->GetPointer() << std::endl;
    ++gitr;
    }

}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
FeatureAggregator<NDimension>
::GenerateData()
{
  this->UpdateAllFeatureGenerators();
  this->ConsolidateFeatures();
}

template <unsigned int NDimension>
unsigned long
FeatureAggregator<NDimension>
::GetMTime() const
{
  // MTime is the max of mtime of all feature generators.
  unsigned long mtime = this->Superclass::GetMTime();
  FeatureGeneratorConstIterator gitr = this->m_FeatureGenerators.begin();
  FeatureGeneratorConstIterator gend = this->m_FeatureGenerators.end();
  while( gitr != gend )
    {
    const unsigned long t = (*gitr)->GetMTime();
    if (t > mtime) 
      {
      mtime = t;
      }
    ++gitr;
    }

  return mtime;
}


/**
 * Update feature generators
 */
template <unsigned int NDimension>
void
FeatureAggregator<NDimension>
::UpdateAllFeatureGenerators()
{
  FeatureGeneratorIterator gitr = this->m_FeatureGenerators.begin();
  FeatureGeneratorIterator gend = this->m_FeatureGenerators.end();

  while( gitr != gend )
    {
    // Assuming that most of the time is spent in generating the features and
    // hardly negligible time is spent in consolidating the features
    this->m_ProgressAccumulator->RegisterInternalFilter( *gitr, 1.0/this->m_FeatureGenerators.size());

    (*gitr)->Update();
    ++gitr;
    }
}

} // end namespace itk

#endif
