/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkConnectedThresholdSegmentationModule.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkConnectedThresholdSegmentationModule_h
#define __itkConnectedThresholdSegmentationModule_h

#include "itkRegionGrowingSegmentationModule.h"
#include "itkConnectedThresholdImageFilter.h"

namespace itk
{

/** \class ConnectedThresholdSegmentationModule
 * \brief This class applies the connected threshold region growing
 * segmentation method.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT ConnectedThresholdSegmentationModule : 
  public RegionGrowingSegmentationModule<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef ConnectedThresholdSegmentationModule              Self;
  typedef RegionGrowingSegmentationModule<NDimension>       Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConnectedThresholdSegmentationModule, RegionGrowingSegmentationModule);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef typename Superclass::FeatureImageType         FeatureImageType;
  typedef typename Superclass::OutputImageType          OutputImageType;
  typedef typename Superclass::InputSpatialObjectType   InputSpatialObjectType;

  /** Upper and Lower thresholds used to control the region growth. */
  itkSetMacro( LowerThreshold, double );
  itkGetMacro( LowerThreshold, double );
  itkSetMacro( UpperThreshold, double );
  itkGetMacro( UpperThreshold, double );

protected:
  ConnectedThresholdSegmentationModule();
  virtual ~ConnectedThresholdSegmentationModule();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  ConnectedThresholdSegmentationModule(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  double        m_LowerThreshold;
  double        m_UpperThreshold;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkConnectedThresholdSegmentationModule.hxx"
#endif

#endif
