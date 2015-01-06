/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFeatureGenerator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFeatureGenerator_hxx
#define __itkFeatureGenerator_hxx

#include "itkFeatureGenerator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
FeatureGenerator<NDimension>
::FeatureGenerator()
{
  this->SetNumberOfRequiredOutputs( 1 );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
FeatureGenerator<NDimension>
::~FeatureGenerator()
{
}

template <unsigned int NDimension>
void
FeatureGenerator<NDimension>
::SetInput( const SpatialObjectType * spatialObject )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<SpatialObjectType *>( spatialObject ));
}

template <unsigned int NDimension>
const typename FeatureGenerator<NDimension>::SpatialObjectType *
FeatureGenerator<NDimension>
::GetFeature() const
{
  if (this->GetNumberOfOutputs() < 1)
    {
    return 0;
    }

  return static_cast<const SpatialObjectType*>(this->ProcessObject::GetOutput(0));

}

template <unsigned int NDimension>
typename FeatureGenerator<NDimension>::SpatialObjectType *
FeatureGenerator<NDimension>
::GetInternalFeature()
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
FeatureGenerator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace itk

#endif
