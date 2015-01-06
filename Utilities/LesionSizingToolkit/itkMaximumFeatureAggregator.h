/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkMaximumFeatureAggregator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMaximumFeatureAggregator_h
#define __itkMaximumFeatureAggregator_h

#include "itkFeatureAggregator.h"

namespace itk
{

/** \class MaximumFeatureAggregator
 * \brief Class for combining multiple features into a single one by computing
 * the pixel-wise maximum. 
 *
 * This class generates a new feature object containing an image that is
 * computed as the pixel-wise maximum of all the input feature images.
 *
 * \warning This class assumes that all the images have the same: origin,
 * spacing, orientation, and that they are represented in the same image grid.
 * mixing strategies.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT MaximumFeatureAggregator : public FeatureAggregator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef MaximumFeatureAggregator            Self;
  typedef FeatureAggregator<NDimension>       Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MaximumFeatureAggregator, FeatureAggregator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of the image and specific SpatialObject produced as output */
  typedef typename Superclass::OutputPixelType                OutputPixelType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::OutputImageSpatialObjectType   OutputImageSpatialObjectType;


protected:
  MaximumFeatureAggregator();
  virtual ~MaximumFeatureAggregator();
  void PrintSelf(std::ostream& os, Indent indent) const;


private:
  MaximumFeatureAggregator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void ConsolidateFeatures();

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkMaximumFeatureAggregator.hxx"
#endif

#endif
