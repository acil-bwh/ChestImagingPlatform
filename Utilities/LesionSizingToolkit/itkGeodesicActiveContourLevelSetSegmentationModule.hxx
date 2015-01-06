/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkGeodesicActiveContourLevelSetSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGeodesicActiveContourLevelSetSegmentationModule_hxx
#define __itkGeodesicActiveContourLevelSetSegmentationModule_hxx

#include "itkGeodesicActiveContourLevelSetSegmentationModule.h"
#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkProgressAccumulator.h"


namespace itk
{


/**
 * Constructor
 */
template <unsigned int NDimension>
GeodesicActiveContourLevelSetSegmentationModule<NDimension>
::GeodesicActiveContourLevelSetSegmentationModule()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
GeodesicActiveContourLevelSetSegmentationModule<NDimension>
::~GeodesicActiveContourLevelSetSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
GeodesicActiveContourLevelSetSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
GeodesicActiveContourLevelSetSegmentationModule<NDimension>
::GenerateData()
{
  typedef GeodesicActiveContourLevelSetImageFilter<
    InputImageType, FeatureImageType, OutputPixelType > FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  filter->SetInput( this->GetInternalInputImage() );
  filter->SetFeatureImage( this->GetInternalFeatureImage() );

  filter->SetMaximumRMSError( this->GetMaximumRMSError() );
  filter->SetNumberOfIterations( this->GetMaximumNumberOfIterations() );
  filter->SetPropagationScaling( this->GetPropagationScaling() );
  filter->SetCurvatureScaling( this->GetCurvatureScaling() );
  filter->SetAdvectionScaling( this->GetAdvectionScaling() );
  filter->UseImageSpacingOn();

  // Progress reporting - forward events from the fast marching filter.
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter( filter, 1.0 );  

  filter->Update();

  std::cout << std::endl;
  std::cout << "Max. no. iterations: " << filter->GetNumberOfIterations() << std::endl;
  std::cout << "Max. RMS error: " << filter->GetMaximumRMSError() << std::endl;
  std::cout << std::endl;
  std::cout << "No. elpased iterations: " << filter->GetElapsedIterations() << std::endl;
  std::cout << "RMS change: " << filter->GetRMSChange() << std::endl;

  this->PackOutputImageInOutputSpatialObject( filter->GetOutput() );
}

} // end namespace itk

#endif
