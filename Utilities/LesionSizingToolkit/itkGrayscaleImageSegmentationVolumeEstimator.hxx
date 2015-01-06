/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkGrayscaleImageSegmentationVolumeEstimator.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGrayscaleImageSegmentationVolumeEstimator_hxx
#define __itkGrayscaleImageSegmentationVolumeEstimator_hxx

#include "itkGrayscaleImageSegmentationVolumeEstimator.h"
#include "itkImageSpatialObject.h"
#include "itkImageRegionIterator.h"


namespace itk
{

/**
 * Constructor
 */
template <unsigned int NDimension>
GrayscaleImageSegmentationVolumeEstimator<NDimension>
::GrayscaleImageSegmentationVolumeEstimator()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
GrayscaleImageSegmentationVolumeEstimator<NDimension>
::~GrayscaleImageSegmentationVolumeEstimator()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
GrayscaleImageSegmentationVolumeEstimator<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/*
 * Generate Data
 */
template <unsigned int NDimension>
void
GrayscaleImageSegmentationVolumeEstimator<NDimension>
::GenerateData()
{
  typename InputImageSpatialObjectType::ConstPointer inputObject =
    dynamic_cast<const InputImageSpatialObjectType * >( this->ProcessObject::GetInput(0) );

  if( !inputObject )
    {
    itkExceptionMacro("Missing input spatial object or incorrect type");
    }

  const InputImageType * inputImage = inputObject->GetImage();

  double sumOfIntensities = 0.0;

  double minimumIntensity = NumericTraits< double >::max();
  double maximumIntensity = NumericTraits< double >::NonpositiveMin();

  typedef ImageRegionConstIterator< InputImageType >  IteratorType;

  typedef typename InputImageType::RegionType   ImageRegionType;

  const ImageRegionType region = inputImage->GetBufferedRegion();

  IteratorType itr( inputImage, region );

  itr.GoToBegin();

  while( !itr.IsAtEnd() )
    {
    const double pixelValue = itr.Get();

    if( pixelValue < minimumIntensity )
      {
      minimumIntensity = pixelValue;
      }

    if( pixelValue > maximumIntensity )
      {
      maximumIntensity = pixelValue;
      }

    sumOfIntensities += pixelValue;
    ++itr;
    }

  const unsigned int long numberOfPixels = region.GetNumberOfPixels();

  sumOfIntensities -= numberOfPixels * minimumIntensity;

  typedef typename InputImageType::SpacingType  SpacingType;

  const SpacingType spacing = inputImage->GetSpacing();

  double pixelVolume = spacing[0] * spacing[1] * spacing[2];

  //
  // Deal with eventual cases of negative spacing
  //
  if( pixelVolume < 0.0 )
    {
    pixelVolume = -pixelVolume;
    }

  const double intensityRange =  (maximumIntensity - minimumIntensity);

  double volumeEstimation = 0.0;

  if( intensityRange > 1e-6 )
    {
    volumeEstimation = pixelVolume * sumOfIntensities / intensityRange;
    }

  RealObjectType * outputCarrier =
    static_cast<RealObjectType*>(this->ProcessObject::GetOutput(0));
 
  outputCarrier->Set( volumeEstimation );

}

} // end namespace itk

#endif
