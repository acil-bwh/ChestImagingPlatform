/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkWeightedSumFeatureAggregator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkWeightedSumFeatureAggregator_h
#define __itkWeightedSumFeatureAggregator_h

#include "itkFeatureAggregator.h"

namespace itk
{

/** \class WeightedSumFeatureAggregator
 * \brief Class for combining multiple features into a single one by computing
 * a pixel-wise weighted sum that is normalized to produce a number between 0
 * and 1.
 *
 * This class generates a new feature object containing an image that is
 * computed as the pixel-wise weighted sum of all the input feature images.
 * The user must provide the weight to be used.
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
class ITK_EXPORT WeightedSumFeatureAggregator : public FeatureAggregator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef WeightedSumFeatureAggregator        Self;
  typedef FeatureAggregator<NDimension>       Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WeightedSumFeatureAggregator, FeatureAggregator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Add the weight that will be used to multiply the corresponding feature
   * when computing the weighted sum. */
  void virtual AddWeight( double weight );

  /** Type of the image and specific SpatialObject produced as output */
  typedef typename Superclass::OutputPixelType                OutputPixelType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::OutputImageSpatialObjectType   OutputImageSpatialObjectType;

protected:
  WeightedSumFeatureAggregator();
  virtual ~WeightedSumFeatureAggregator();
  void PrintSelf(std::ostream& os, Indent indent) const;


private:
  WeightedSumFeatureAggregator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void ConsolidateFeatures();

  typedef std::vector< double >     WeightsArrayType;

  WeightsArrayType                  m_Weights;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkWeightedSumFeatureAggregator.hxx"
#endif

#endif
