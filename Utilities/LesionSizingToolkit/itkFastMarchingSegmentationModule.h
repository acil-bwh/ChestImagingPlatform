/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFastMarchingSegmentationModule.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFastMarchingSegmentationModule_h
#define __itkFastMarchingSegmentationModule_h

#include "itkSinglePhaseLevelSetSegmentationModule.h"
#include "itkImageSpatialObject.h"
#include "itkLandmarkSpatialObject.h"

namespace itk
{

/** \class FastMarchingSegmentationModule
 * \brief Class applies a fast marching segmentation method
 *
 * Takes as input a landmark spatial object and a feature image and produces as
 * output a segmentation of the output level set. Threshold this at 0 and you
 * will get the zero set. 
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT FastMarchingSegmentationModule : public SinglePhaseLevelSetSegmentationModule<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef FastMarchingSegmentationModule                        Self;
  typedef SinglePhaseLevelSetSegmentationModule<NDimension>     Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FastMarchingSegmentationModule, SinglePhaseLevelSetSegmentationModule);

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

  /** Set the Fast Marching algorithm Stopping Value. The Fast Marching
   * algorithm is terminated when the value of the smallest trial point
   * is greater than the stopping value. */
  itkSetMacro( StoppingValue, double );
  itkGetMacro( StoppingValue, double );

  /** Set the Fast Marching algorithm distance from seeds. */
  itkSetMacro( DistanceFromSeeds, double );
  itkGetMacro( DistanceFromSeeds, double );

protected:
  FastMarchingSegmentationModule();
  virtual ~FastMarchingSegmentationModule();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData () override;

  /** Extract the input set of landmark points to be used as seeds. */
  const InputSpatialObjectType * GetInternalInputLandmarks() const;

  double m_StoppingValue;
  double m_DistanceFromSeeds;

private:
  FastMarchingSegmentationModule(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkFastMarchingSegmentationModule.hxx"
#endif

#endif
