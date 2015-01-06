/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSegmentationVolumeEstimator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSegmentationVolumeEstimator_hxx
#define __itkSegmentationVolumeEstimator_hxx

#include "itkSegmentationVolumeEstimator.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
SegmentationVolumeEstimator<NDimension>
::SegmentationVolumeEstimator()
{
  this->SetNumberOfRequiredInputs( 1 );   // for the segmentation Spatial Object
  this->SetNumberOfRequiredOutputs( 1 );  // for the Volume

  RealObjectType::Pointer output = RealObjectType::New();
  this->ProcessObject::SetNthOutput( 0, output.GetPointer() );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
SegmentationVolumeEstimator<NDimension>
::~SegmentationVolumeEstimator()
{
}


/**
 * Return the value of the estimated volume
 */
template <unsigned int NDimension>
typename SegmentationVolumeEstimator<NDimension>::RealType 
SegmentationVolumeEstimator<NDimension>
::GetVolume() const
{
  return this->GetVolumeOutput()->Get();
}


/**
 * Return the value of the estimated volume stored in a DataObject decorator
 * that can be passed down a pipeline.
 */
template <unsigned int NDimension>
const typename SegmentationVolumeEstimator<NDimension>::RealObjectType * 
SegmentationVolumeEstimator<NDimension>
::GetVolumeOutput() const
{
  return static_cast<const RealObjectType*>(this->ProcessObject::GetOutput(0));
}


/**
 * Set the input SpatialObject representing the segmentation whose volume will
 * be estimated by this class.
 */
template <unsigned int NDimension>
void
SegmentationVolumeEstimator<NDimension>
::SetInput( const SpatialObjectType * inputSpatialObject )
{
  this->SetNthInput(0, const_cast<SpatialObjectType *>( inputSpatialObject ));
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
SegmentationVolumeEstimator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
SegmentationVolumeEstimator<NDimension>
::GenerateData()
{
  // This method is intended to be overridden by derived classes
}

} // end namespace itk

#endif
