/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLesionSegmentationMethod.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLesionSegmentationMethod_h
#define __itkLesionSegmentationMethod_h

#include "itkProcessObject.h"
#include "itkImage.h"
#include "itkDataObjectDecorator.h"
#include "itkSpatialObject.h"
#include "itkFeatureGenerator.h"
#include "itkSegmentationModule.h"
#include "itkProgressAccumulator.h"

namespace itk
{

/** \class LesionSegmentationMethod
 * \brief Class for coordinating components of a segmentation framework.
 *
 * This class was designed to facilitate the segmentation of lung lesions, but
 * yet to be useful for other segmentation tasks.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT LesionSegmentationMethod : public ProcessObject
{
public:
  /** Standard class typedefs. */
  typedef LesionSegmentationMethod      Self;
  typedef ProcessObject                 Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LesionSegmentationMethod, ProcessObject);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input and output of this
   * segmentation method. */
  typedef SpatialObject< NDimension >                 SpatialObjectType;
  typedef typename SpatialObjectType::Pointer         SpatialObjectPointer;
  typedef typename SpatialObjectType::ConstPointer    SpatialObjectConstPointer;

  /** SpatialObject that defines the Region of Interest in the input data */
  itkSetObjectMacro( RegionOfInterest, SpatialObjectType );
  itkGetConstObjectMacro( RegionOfInterest, SpatialObjectType );

  /** SpatialObject that defines the initial segmentation. This will be
   * used to initialize the segmentation process driven by the
   * LesionSegmentationMethod. */
  itkSetConstObjectMacro( InitialSegmentation, SpatialObjectType );
  itkGetConstObjectMacro( InitialSegmentation, SpatialObjectType );

  /** Type of the class that will generate input features in the form of
   * spatial objects. */
  typedef FeatureGenerator< Dimension >         FeatureGeneratorType;

  /**
   * Method for adding a feature generator that will compute the Nth feature to
   * be passed to the segmentation module.
   */
  void AddFeatureGenerator( FeatureGeneratorType * generator ); 

  /** Type of the segmentation module that encapsulate the actual segmentation
   * algorithm. */
  typedef SegmentationModule< Dimension >             SegmentationModuleType;
  typedef typename SegmentationModuleType::Pointer    SegmentationModulePointer;


  /**
   * Method for setting the class that encapsulates the actual segmentation
   * algorithm.
   */
  itkSetObjectMacro( SegmentationModule, SegmentationModuleType );


protected:
  LesionSegmentationMethod();
  virtual ~LesionSegmentationMethod();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData() override;

private:
  LesionSegmentationMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  SpatialObjectConstPointer                 m_RegionOfInterest;
  SpatialObjectConstPointer                 m_InitialSegmentation;
  
  typedef typename FeatureGeneratorType::Pointer                FeatureGeneratorPointer;
  typedef std::vector< FeatureGeneratorPointer >                FeatureGeneratorArrayType;
  typedef typename FeatureGeneratorArrayType::iterator          FeatureGeneratorIterator;
  typedef typename FeatureGeneratorArrayType::const_iterator    FeatureGeneratorConstIterator;

  FeatureGeneratorArrayType                 m_FeatureGenerators;

  SegmentationModulePointer                 m_SegmentationModule;

  ProgressAccumulator::Pointer              m_ProgressAccumulator;


  /** This method calls the Update() method of each one of the feature generators */
  void UpdateAllFeatureGenerators();

  /** This method compares the number of available features against the number
   * of expected features, and it will throw an exception if they do not match.
   */
  void VerifyNumberOfAvailableFeaturesMatchedExpectations() const;

  /** Connect the outputs of feature generators as input to the segmentation module */
  void ConnectFeaturesToSegmentationModule();

  void ExecuteSegmentationModule();

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkLesionSegmentationMethod.hxx"
#endif

#endif
