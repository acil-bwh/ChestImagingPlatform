/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkGrayscaleImageSegmentationVolumeEstimator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGrayscaleImageSegmentationVolumeEstimator_h
#define __itkGrayscaleImageSegmentationVolumeEstimator_h

#include "itkSegmentationVolumeEstimator.h"

namespace itk
{

/** \class GrayscaleImageSegmentationVolumeEstimator
 * \brief Class for estimating the volume of a segmentation stored in a
 * SpatialObject that carries a gray-scale image of pixel type float. This
 * implementation assumes that the input image is roughly composed of two
 * regions with intensity plateaus, and with a narrow transition region between
 * the two regions.  Note that this doesn't mean that the regions must be a
 * single connected component.
 *
 * The estimation of the volume is done by the equivalent of rescaling the
 * intensity range to [0:1] and then adding the contributions of all the
 * pixels. 
 *
 * The pixels size is, of course, taken into account.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT GrayscaleImageSegmentationVolumeEstimator :
 public SegmentationVolumeEstimator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef GrayscaleImageSegmentationVolumeEstimator   Self;
  typedef SegmentationVolumeEstimator<NDimension>     Superclass;
  typedef SmartPointer<Self>                          Pointer;
  typedef SmartPointer<const Self>                    ConstPointer;
  typedef typename Superclass::RealObjectType         RealObjectType;

  /** Method for constructing new instances of this class. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(GrayscaleImageSegmentationVolumeEstimator, SegmentationVolumeEstimator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef typename Superclass::SpatialObjectType          SpatialObjectType;
  typedef typename Superclass::SpatialObjectPointer       SpatialObjectPointer;
  typedef typename Superclass::SpatialObjectConstPointer  SpatialObjectConstPointer;

  /** Required type of the input */
  typedef float                                               InputPixelType;
  typedef ImageSpatialObject< NDimension, InputPixelType >    InputImageSpatialObjectType;
  typedef Image< InputPixelType, NDimension >                 InputImageType;

protected:
  GrayscaleImageSegmentationVolumeEstimator();
  virtual ~GrayscaleImageSegmentationVolumeEstimator();

  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData();

private:
  GrayscaleImageSegmentationVolumeEstimator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkGrayscaleImageSegmentationVolumeEstimator.hxx"
#endif

#endif
