/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkShapeDetectionLevelSetSegmentationModule.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkShapeDetectionLevelSetSegmentationModule_hxx
#define __itkShapeDetectionLevelSetSegmentationModule_hxx

#include "itkShapeDetectionLevelSetSegmentationModule.h"
#include "itkShapeDetectionLevelSetImageFilter.h"


namespace itk
{


/**
 * Constructor
 */
template <unsigned int NDimension>
ShapeDetectionLevelSetSegmentationModule<NDimension>
::ShapeDetectionLevelSetSegmentationModule()
{
}


/**
 * Destructor
 */
template <unsigned int NDimension>
ShapeDetectionLevelSetSegmentationModule<NDimension>
::~ShapeDetectionLevelSetSegmentationModule()
{
}


/**
 * PrintSelf
 */
template <unsigned int NDimension>
void
ShapeDetectionLevelSetSegmentationModule<NDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}


/**
 * Generate Data
 */
template <unsigned int NDimension>
void
ShapeDetectionLevelSetSegmentationModule<NDimension>
::GenerateData()
{
  typedef ShapeDetectionLevelSetImageFilter<
    InputImageType, FeatureImageType, OutputPixelType > FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  filter->SetInput( this->GetInternalInputImage() );
  filter->SetIsoSurfaceValue( 0.0 ); // Zero Set value
  filter->SetFeatureImage( this->GetInternalFeatureImage() );

  filter->SetMaximumRMSError( this->GetMaximumRMSError() );
  filter->SetNumberOfIterations( this->GetMaximumNumberOfIterations() );
  filter->SetPropagationScaling( this->GetPropagationScaling() );
  filter->SetCurvatureScaling( this->GetCurvatureScaling() );
  filter->SetAdvectionScaling( 0.0 );
  filter->UseImageSpacingOn();

  std::cout << "Propagation Scaling = " << this->GetPropagationScaling() << std::endl;
  std::cout << "Curvature Scaling = " << this->GetCurvatureScaling() << std::endl;

  filter->Update();

  std::cout << "Max. no. iterations: " << filter->GetNumberOfIterations() << std::endl;
  std::cout << "Max. RMS error: " << filter->GetMaximumRMSError() << std::endl;
  std::cout << "No. elpased iterations: " << filter->GetElapsedIterations() << std::endl;
  std::cout << "RMS change: " << filter->GetRMSChange() << std::endl;

  this->PackOutputImageInOutputSpatialObject( filter->GetOutput() );
}

} // end namespace itk

#endif
