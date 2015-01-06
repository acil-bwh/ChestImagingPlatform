/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkMinimumFeatureAggregator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMinimumFeatureAggregator_hxx
#define __itkMinimumFeatureAggregator_hxx

#include "itkMinimumFeatureAggregator.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
MinimumFeatureAggregator<NDimension>
::MinimumFeatureAggregator()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
MinimumFeatureAggregator<NDimension>
::~MinimumFeatureAggregator()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
MinimumFeatureAggregator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

 
template <unsigned int NDimension>
void
MinimumFeatureAggregator<NDimension>
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
  consolidatedFeatureImage->FillBuffer( NumericTraits< FeaturePixelType >::max() );

  const unsigned int numberOfFeatures = this->GetNumberOfInputFeatures();

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
   
    while( !srcitr.IsAtEnd() )
      {
      if( dstitr.Get() > srcitr.Get() )
        {
        dstitr.Set( srcitr.Get() );
        }
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
