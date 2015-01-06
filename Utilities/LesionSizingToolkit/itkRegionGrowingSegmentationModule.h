/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkRegionGrowingSegmentationModule.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkRegionGrowingSegmentationModule_h
#define __itkRegionGrowingSegmentationModule_h

#include "itkSegmentationModule.h"
#include "itkImageSpatialObject.h"
#include "itkLandmarkSpatialObject.h"

namespace itk
{

/** \class RegionGrowingSegmentationModule
 * \brief Class applies a region growing segmentation method
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT RegionGrowingSegmentationModule : public SegmentationModule<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef RegionGrowingSegmentationModule       Self;
  typedef SegmentationModule<NDimension>        Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RegionGrowingSegmentationModule, SegmentationModule);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef typename Superclass::SpatialObjectType         SpatialObjectType;
  typedef typename Superclass::SpatialObjectPointer      SpatialObjectPointer;

  /** Types of the input feature image and the output image */
  typedef float                                         OutputPixelType;
  typedef float                                         FeaturePixelType;
  typedef Image< FeaturePixelType, NDimension >         FeatureImageType;
  typedef Image< OutputPixelType, NDimension >          OutputImageType;

  /** Types of the Spatial objects used for the input feature image and the output image. */
  typedef ImageSpatialObject< NDimension, FeaturePixelType >   FeatureSpatialObjectType;
  typedef ImageSpatialObject< NDimension, OutputPixelType >    OutputSpatialObjectType;

  /** Type of the input set of seed points. They are stored in a Landmark Spatial Object. */
  typedef LandmarkSpatialObject< NDimension >                  InputSpatialObjectType;

protected:
  RegionGrowingSegmentationModule();
  virtual ~RegionGrowingSegmentationModule();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

  /** Set the output image as cargo of the output SpatialObject. */
  void PackOutputImageInOutputSpatialObject( OutputImageType * outputImage );

  /** Extract the input set of landmark points to be used as seeds. */
  const InputSpatialObjectType * GetInternalInputLandmarks() const;

  /** Extract the input feature image from the input feature spatial object. */
  const FeatureImageType * GetInternalFeatureImage() const;

private:
  RegionGrowingSegmentationModule(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void ConvertIntensitiesToCenteredRange( OutputImageType * outputImage );
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkRegionGrowingSegmentationModule.hxx"
#endif

#endif
