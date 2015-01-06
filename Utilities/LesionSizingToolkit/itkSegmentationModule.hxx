/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSegmentationModule_hxx
#define __itkSegmentationModule_hxx

#include "itkSegmentationModule.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
SegmentationModule<NDimension>
::SegmentationModule()
{
  this->SetNumberOfRequiredOutputs( 1 );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
SegmentationModule<NDimension>
::~SegmentationModule()
{
}

template <unsigned int NDimension>
void
SegmentationModule<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename SegmentationModule<NDimension>::SpatialObjectType *
SegmentationModule<NDimension>
::GetInput() const
{
  // Process object is not const-correct so the const casting is required.
  const SpatialObjectType * input =
    dynamic_cast<const SpatialObjectType *>( this->ProcessObject::GetInput(0) );
  return input;
}


template <unsigned int NDimension>
void
SegmentationModule<NDimension>
::SetFeature( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(1, const_cast<SpatialObjectType *>( spatialObject ));
}


template <unsigned int NDimension>
const typename SegmentationModule<NDimension>::SpatialObjectType *
SegmentationModule<NDimension>
::GetFeature() const
{
  // Process object is not const-correct so the const casting is required.
  const SpatialObjectType * feature =
    dynamic_cast<const SpatialObjectType *>( this->ProcessObject::GetInput(1) );
  return feature;
}


template <unsigned int NDimension>
unsigned int
SegmentationModule<NDimension>
::GetExpectedNumberOfFeatures() const
{
  return 1;
}


template <unsigned int NDimension>
const typename SegmentationModule<NDimension>::SpatialObjectType *
SegmentationModule<NDimension>
::GetOutput() const
{
  if (this->GetNumberOfOutputs() < 1)
    {
    return 0;
    }

  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}


/** This non-const version is intended only for the internal use in the derived
 * classes. */
template <unsigned int NDimension>
typename SegmentationModule<NDimension>::SpatialObjectType *
SegmentationModule<NDimension>
::GetInternalOutput()
{
  if (this->GetNumberOfOutputs() < 1)
    {
    return 0;
    }

  return static_cast<SpatialObjectType*>(this->ProcessObject::GetOutput(0));
}


/*
 * PrintSelf
 */
template <unsigned int NDimension>
void
SegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace itk

#endif
