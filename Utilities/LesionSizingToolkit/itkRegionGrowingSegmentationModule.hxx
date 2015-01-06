/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkRegionGrowingSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkRegionGrowingSegmentationModule_hxx
#define __itkRegionGrowingSegmentationModule_hxx

#include "itkRegionGrowingSegmentationModule.h"
#include "itkImageRegionIterator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
RegionGrowingSegmentationModule<NDimension>
::RegionGrowingSegmentationModule()
{
  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 1 );

  typename OutputSpatialObjectType::Pointer outputObject = OutputSpatialObjectType::New();

  this->ProcessObject::SetNthOutput( 0, outputObject.GetPointer() );
}


/**
 * Destructor
 */
template <unsigned int NDimension>
RegionGrowingSegmentationModule<NDimension>
::~RegionGrowingSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
RegionGrowingSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
RegionGrowingSegmentationModule<NDimension>
::GenerateData()
{

}


/**
 * This method is intended to be used only by the subclasses to extract the
 * input image from the input SpatialObject.
 */
template <unsigned int NDimension>
const typename RegionGrowingSegmentationModule<NDimension>::InputSpatialObjectType *
RegionGrowingSegmentationModule<NDimension>
::GetInternalInputLandmarks() const
{
  const InputSpatialObjectType * inputObject =
    dynamic_cast< const InputSpatialObjectType * >( this->GetInput() );

  return inputObject;
}


/**
 * This method is intended to be used only by the subclasses to extract the
 * input feature image from the input feature SpatialObject.
 */
template <unsigned int NDimension>
const typename RegionGrowingSegmentationModule<NDimension>::FeatureImageType *
RegionGrowingSegmentationModule<NDimension>
::GetInternalFeatureImage() const
{
  const FeatureSpatialObjectType * featureObject =
    dynamic_cast< const FeatureSpatialObjectType * >( this->GetFeature() );

  const FeatureImageType * featureImage = featureObject->GetImage();

  return featureImage;
}


/**
 * This method is intended to be used only by the subclasses to insert the
 * output image as cargo of the output spatial object.
 */
template <unsigned int NDimension>
void
RegionGrowingSegmentationModule<NDimension>
::PackOutputImageInOutputSpatialObject( OutputImageType * image )
{
  typename OutputImageType::Pointer outputImage = image;

  outputImage->DisconnectPipeline();

  this->ConvertIntensitiesToCenteredRange( outputImage );

  OutputSpatialObjectType * outputObject =
    dynamic_cast< OutputSpatialObjectType * >(this->ProcessObject::GetOutput(0));

  outputObject->SetImage( outputImage );
}

/**
 * This method is intended to be used only by this class. It should be called
 * from the PackOutputImageInOutputSpatialObject() method.
 */
template <unsigned int NDimension>
void
RegionGrowingSegmentationModule<NDimension>
::ConvertIntensitiesToCenteredRange( OutputImageType * image )
{
  typedef ImageRegionIterator< OutputImageType > IteratorType;

  IteratorType itr( image, image->GetBufferedRegion() );
  
  itr.GoToBegin();

  //
  // Convert intensities to centered range
  //
  while( !itr.IsAtEnd() )
    {
    if( itr.Get() )
      {
      itr.Set( 4.0 );
      }
    else
      {
      itr.Set( -4.0 );
      }
    ++itr;
    }
}


} // end namespace itk

#endif
