/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkFeatureAggregator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFeatureAggregator_h
#define __itkFeatureAggregator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkProgressAccumulator.h"

namespace itk
{

/** \class FeatureAggregator
 * \brief Class for combining multiple features into a single one. 
 *
 * This class is the base class for specific implementation of feature
 * mixing strategies.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT FeatureAggregator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef FeatureAggregator             Self;
  typedef FeatureGenerator<NDimension>  Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  /** This is an abstract class, therefore it doesn't need the itkNewMacro() */

  /** Run-time type information (and related methods). */
  itkTypeMacro(FeatureAggregator, FeatureGenerator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef SpatialObject< NDimension >                 SpatialObjectType;
  typedef typename SpatialObjectType::Pointer         SpatialObjectPointer;
  typedef typename SpatialObjectType::ConstPointer    SpatialObjectConstPointer;

  /** Type of the image and specific SpatialObject produced as output */
  typedef float                                                 OutputPixelType;
  typedef Image< OutputPixelType, NDimension >                  OutputImageType;
  typedef ImageSpatialObject< NDimension, OutputPixelType >     OutputImageSpatialObjectType;

  /** Type of the class that will generate input features in the form of
   * spatial objects. */
  typedef FeatureGenerator< Dimension >         FeatureGeneratorType;

  /**
   * Method for adding a feature generator that will compute the Nth feature to
   * be passed to the segmentation module.
   */
  void AddFeatureGenerator( FeatureGeneratorType * generator ); 

  /** Check all feature generators and return consolidate MTime */
  virtual unsigned long GetMTime() const override;

protected:
  FeatureAggregator();
  virtual ~FeatureAggregator();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData() override;

  unsigned int GetNumberOfInputFeatures() const;

  typedef typename FeatureGeneratorType::SpatialObjectType      InputFeatureType;

  const InputFeatureType * GetInputFeature( unsigned int featureId ) const;

  ProgressAccumulator::Pointer              m_ProgressAccumulator;

private:
  FeatureAggregator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef typename FeatureGeneratorType::Pointer                FeatureGeneratorPointer;
  typedef std::vector< FeatureGeneratorPointer >                FeatureGeneratorArrayType;
  typedef typename FeatureGeneratorArrayType::iterator          FeatureGeneratorIterator;
  typedef typename FeatureGeneratorArrayType::const_iterator    FeatureGeneratorConstIterator;

  FeatureGeneratorArrayType                 m_FeatureGenerators;

  void UpdateAllFeatureGenerators();

  void virtual ConsolidateFeatures() = 0;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkFeatureAggregator.hxx"
#endif

#endif
