/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkConfidenceConnectedSegmentationModule.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkConfidenceConnectedSegmentationModule_h
#define __itkConfidenceConnectedSegmentationModule_h

#include "itkRegionGrowingSegmentationModule.h"
#include "itkConfidenceConnectedImageFilter.h"

namespace itk
{

/** \class ConfidenceConnectedSegmentationModule
 * \brief This class applies the connected threshold region growing
 * segmentation method.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT ConfidenceConnectedSegmentationModule : 
  public RegionGrowingSegmentationModule<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef ConfidenceConnectedSegmentationModule             Self;
  typedef RegionGrowingSegmentationModule<NDimension>       Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConfidenceConnectedSegmentationModule, RegionGrowingSegmentationModule);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef typename Superclass::FeatureImageType         FeatureImageType;
  typedef typename Superclass::OutputImageType          OutputImageType;
  typedef typename Superclass::InputSpatialObjectType   InputSpatialObjectType;

  /** Factor that will be applied to the standard deviation in order to compute
   * the intensity range from which pixel will be included in the region. */
  itkSetMacro( SigmaMultiplier, double );
  itkGetMacro( SigmaMultiplier, double );

protected:
  ConfidenceConnectedSegmentationModule();
  virtual ~ConfidenceConnectedSegmentationModule();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  ConfidenceConnectedSegmentationModule(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  double        m_SigmaMultiplier;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkConfidenceConnectedSegmentationModule.hxx"
#endif

#endif
