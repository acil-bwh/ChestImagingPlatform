/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFastMarchingSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFastMarchingSegmentationModule_hxx
#define __itkFastMarchingSegmentationModule_hxx

#include "itkFastMarchingSegmentationModule.h"
#include "itkImageRegionIterator.h"
#include "itkFastMarchingImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkProgressAccumulator.h"

namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
FastMarchingSegmentationModule<NDimension>
::FastMarchingSegmentationModule()
{
  this->m_StoppingValue = static_cast<double>( static_cast<OutputPixelType>( 
                      NumericTraits<OutputPixelType>::max() / 2.0 ) );

  this->m_DistanceFromSeeds = 0.0;
  
  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 1 );

  typename OutputSpatialObjectType::Pointer outputObject = OutputSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FastMarchingSegmentationModule<NDimension>
::~FastMarchingSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
FastMarchingSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Stopping Value = " << this->m_StoppingValue << std::endl;
  os << indent << "Distance from seeds = " << this->m_DistanceFromSeeds << std::endl;
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
FastMarchingSegmentationModule<NDimension>
::GenerateData()
{
  typedef FastMarchingImageFilter< FeatureImageType, OutputImageType >      FilterType;
  
  typename FilterType::Pointer filter = FilterType::New();

  const FeatureImageType * featureImage = this->GetInternalFeatureImage();

  filter->SetInput( featureImage );

  filter->SetStoppingValue( this->m_StoppingValue );

  // Progress reporting - forward events from the fast marching filter.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( filter, 0.9 );
  
  const InputSpatialObjectType * inputSeeds = this->GetInternalInputLandmarks();
  const unsigned int numberOfPoints = inputSeeds->GetNumberOfPoints();

  typedef typename InputSpatialObjectType::SpatialObjectPointType   SpatialObjectPointType;
  typedef typename SpatialObjectPointType::PointType                PointType;
  typedef typename InputSpatialObjectType::LandmarkPointListType    LandmarkPointListType;
  typedef typename FeatureImageType::IndexType                      IndexType;
  typedef typename FeatureImageType::IndexType                      IndexType;
  typedef typename FilterType::NodeContainer                        NodeContainer;
  typedef typename FilterType::NodeType                             NodeType;

  typename NodeContainer::Pointer trialPoints = NodeContainer::New();
  
  const LandmarkPointListType & points = inputSeeds->GetPoints();

  IndexType index;

  for( unsigned int i=0; i < numberOfPoints; i++ )
    {
    featureImage->TransformPhysicalPointToIndex( points[i].GetPositionInWorldSpace(), index );

    NodeType node;

    // By starting the FastMarching front at this value,
    // the zero set will end up being placed at distance
    // = value from the seeds. That can be seen as computing
    // a distance map from the seeds.
    node.SetValue( -this->m_DistanceFromSeeds );
    
    node.SetIndex( index );
    trialPoints->InsertElement( i, node );  
    }

  filter->SetTrialPoints( trialPoints );
  filter->Update();

  // Rescale the values to make the output intensity fit in the expected
  // range of [-4:4]
  typedef itk::IntensityWindowingImageFilter<  OutputImageType, OutputImageType > WindowingFilterType;
  typename WindowingFilterType::Pointer windowing = WindowingFilterType::New();
  windowing->SetInput( filter->GetOutput() );
  windowing->SetWindowMinimum( -this->m_DistanceFromSeeds );
  windowing->SetWindowMaximum(  this->m_StoppingValue );
  windowing->SetOutputMinimum( -4.0 );
  windowing->SetOutputMaximum(  4.0 );
  windowing->InPlaceOn();
  progress->RegisterInternalFilter( windowing, 0.1 );  
  windowing->Update();

  this->PackOutputImageInOutputSpatialObject( windowing->GetOutput() );
}


/**
 * This method is intended to be used only by the subclasses to extract the
 * input image from the input SpatialObject.
 */
template <unsigned int NDimension>
const typename FastMarchingSegmentationModule<NDimension>::InputSpatialObjectType *
FastMarchingSegmentationModule<NDimension>
::GetInternalInputLandmarks() const
{
  const InputSpatialObjectType * inputObject =
    dynamic_cast< const InputSpatialObjectType * >( this->GetInput() );

  return inputObject;
}

} // end namespace itk

#endif
