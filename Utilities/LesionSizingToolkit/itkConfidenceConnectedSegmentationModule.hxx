/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkConfidenceConnectedSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkConfidenceConnectedSegmentationModule_hxx
#define __itkConfidenceConnectedSegmentationModule_hxx

#include "itkConfidenceConnectedSegmentationModule.h"
#include "itkProgressAccumulator.h"


namespace itk
{


/**
 * Constructor
 */
template <unsigned int NDimension>
ConfidenceConnectedSegmentationModule<NDimension>
::ConfidenceConnectedSegmentationModule()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
ConfidenceConnectedSegmentationModule<NDimension>
::~ConfidenceConnectedSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
ConfidenceConnectedSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
ConfidenceConnectedSegmentationModule<NDimension>
::GenerateData()
{
  typedef ConfidenceConnectedImageFilter<
    FeatureImageType, OutputImageType >           FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  const FeatureImageType * featureImage = this->GetInternalFeatureImage();

  filter->SetInput( featureImage );
  
  const InputSpatialObjectType * inputSeeds = this->GetInternalInputLandmarks();
 
  const unsigned int numberOfPoints = inputSeeds->GetNumberOfPoints();

  typedef typename InputSpatialObjectType::SpatialObjectPointType   SpatialObjectPointType;
  typedef typename SpatialObjectPointType::PointType                PointType;
  typedef typename InputSpatialObjectType::PointListType            PointListType;
  typedef typename FeatureImageType::IndexType                      IndexType;

  const PointListType & points = inputSeeds->GetPoints();

  IndexType index;

  for( unsigned int i=0; i < numberOfPoints; i++ )
    {
    featureImage->TransformPhysicalPointToIndex( points[i].GetPosition(), index );
    filter->AddSeed( index );
    }

  filter->SetMultiplier( this->m_SigmaMultiplier );

  filter->SetReplaceValue( 1.0 );
  filter->SetNumberOfIterations( 5 );
  filter->SetInitialNeighborhoodRadius( 2 );

  // Report progress.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( filter, 1.0 );

  filter->Update();

  this->PackOutputImageInOutputSpatialObject( filter->GetOutput() );
}

} // end namespace itk

#endif
