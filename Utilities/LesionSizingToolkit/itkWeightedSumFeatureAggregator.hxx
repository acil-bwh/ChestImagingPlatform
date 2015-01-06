/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkWeightedSumFeatureAggregator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkWeightedSumFeatureAggregator_hxx
#define __itkWeightedSumFeatureAggregator_hxx

#include "itkWeightedSumFeatureAggregator.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
WeightedSumFeatureAggregator<NDimension>
::WeightedSumFeatureAggregator()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
WeightedSumFeatureAggregator<NDimension>
::~WeightedSumFeatureAggregator()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
WeightedSumFeatureAggregator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

  
template <unsigned int NDimension>
void
WeightedSumFeatureAggregator<NDimension>
::AddWeight( double weight )
{
  this->m_Weights.push_back( weight );
  this->Modified();
}


template <unsigned int NDimension>
void
WeightedSumFeatureAggregator<NDimension>
::ConsolidateFeatures()
{
  typedef float                                                   FeaturePixelType;
  typedef Image< FeaturePixelType, NDimension >                   FeatureImageType;
  typedef ImageSpatialObject< NDimension, FeaturePixelType >      FeatureSpatialObjectType;

  const FeatureSpatialObjectType * firstFeatureObject =
    dynamic_cast< const FeatureSpatialObjectType * >( this->GetInputFeature(0) );

  const FeatureImageType * firstFeatureImage = firstFeatureObject->GetImage();

  typename FeatureImageType::Pointer consolidatedFeatureImage = FeatureImageType::New();

  consolidatedFeatureImage->CopyInformation( firstFeatureImage );
  consolidatedFeatureImage->SetRegions( firstFeatureImage->GetBufferedRegion() );
  consolidatedFeatureImage->Allocate();
  consolidatedFeatureImage->FillBuffer( NumericTraits< FeaturePixelType >::ZeroValue() );

  const unsigned int numberOfFeatures = this->GetNumberOfInputFeatures();

  const unsigned int numberOfWeights = this->m_Weights.size();

  if( numberOfFeatures != numberOfWeights )
    {
    itkExceptionMacro("Number of Weights " << numberOfWeights 
      << " different from " << " number of Features " << numberOfFeatures );
    }

  double sumOfWeights = 0.0;

  for( unsigned int k = 0; k < numberOfWeights; k++ )
    {
    sumOfWeights += this->m_Weights[k]; 
    }

  for( unsigned int i = 0; i < numberOfFeatures; i++ )
    {
    const FeatureSpatialObjectType * featureObject =
      dynamic_cast< const FeatureSpatialObjectType * >( this->GetInputFeature(i) );

    const FeatureImageType * featureImage = featureObject->GetImage();

    typedef ImageRegionIterator< FeatureImageType >          FeatureIterator;
    typedef ImageRegionConstIterator< FeatureImageType >     FeatureConstIterator;

    FeatureIterator       dstitr( consolidatedFeatureImage, consolidatedFeatureImage->GetBufferedRegion() );
    FeatureConstIterator  srcitr( featureImage, featureImage->GetBufferedRegion() );

    dstitr.GoToBegin();
    srcitr.GoToBegin();
   
    const double weight = this->m_Weights[i] / sumOfWeights;

    while( !srcitr.IsAtEnd() )
      {
      dstitr.Set( dstitr.Get() + srcitr.Get() * weight );
      ++srcitr;
      ++dstitr;
      }
    }

  FeatureSpatialObjectType * outputObject =
    dynamic_cast< FeatureSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( consolidatedFeatureImage );
}

} // end namespace itk

#endif
