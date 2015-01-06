/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkMorphologicalOpenningFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMorphologicalOpenningFeatureGenerator_h
#define __itkMorphologicalOpenningFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkVotingBinaryHoleFillFloodingImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkCastImageFilter.h"

namespace itk
{

/** \class MorphologicalOpenningFeatureGenerator
 * \brief Generates a feature image based on intensity and removes small pieces from it.
 *
 * This feature generator thresholds the input image, runs an Openning
 * Mathematical Morphology Filter and then a Voting Hole Filling filter.
 * The net effect is the elimination of small islands and small holes 
 * from the thresholded image.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT MorphologicalOpenningFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef MorphologicalOpenningFeatureGenerator         Self;
  typedef FeatureGenerator<NDimension>                  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MorphologicalOpenningFeatureGenerator, FeatureGenerator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input to this
   * feature generator. */
  typedef signed short                                      InputPixelType;
  typedef Image< InputPixelType, Dimension >                InputImageType;
  typedef ImageSpatialObject< NDimension, InputPixelType >  InputImageSpatialObjectType;
  typedef typename InputImageSpatialObjectType::Pointer     InputImageSpatialObjectPointer;
  typedef typename Superclass::SpatialObjectType            SpatialObjectType;

  /** Input data that will be used for generating the feature. */
  using ProcessObject::SetInput;
  void SetInput( const SpatialObjectType * input );
  const SpatialObjectType * GetInput() const;

  /** Output data that carries the feature in the form of a
   * SpatialObject. */
  const SpatialObjectType * GetFeature() const;

  /** Set the Hounsfield Unit value to threshold the Lung. */
  itkSetMacro( LungThreshold, InputPixelType );
  itkGetMacro( LungThreshold, InputPixelType );

protected:
  MorphologicalOpenningFeatureGenerator();
  virtual ~MorphologicalOpenningFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  MorphologicalOpenningFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef unsigned char                               InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef float                                       OutputPixelType;
  typedef Image< OutputPixelType, Dimension >         OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef BinaryThresholdImageFilter<
    InputImageType, InternalImageType >                   ThresholdFilterType;
  typedef typename ThresholdFilterType::Pointer           ThresholdFilterPointer;

  typedef BinaryBallStructuringElement< InternalPixelType, Dimension > KernelType;
  typedef BinaryMorphologicalOpeningImageFilter< 
    InternalImageType, InternalImageType, KernelType >    OpenningFilterType;
  typedef typename OpenningFilterType::Pointer            OpenningFilterPointer;

  typedef VotingBinaryHoleFillFloodingImageFilter<
    InternalImageType, InternalImageType >                VotingHoleFillingFilterType;
  typedef typename VotingHoleFillingFilterType::Pointer   VotingHoleFillingFilterPointer;

  typedef CastImageFilter<
    InternalImageType, OutputImageType >                  CastingFilterType;
  typedef typename CastingFilterType::Pointer             CastingFilterPointer;


  ThresholdFilterPointer                m_ThresholdFilter;
  OpenningFilterPointer                 m_OpenningFilter;
  VotingHoleFillingFilterPointer        m_VotingHoleFillingFilter;
  CastingFilterPointer                  m_CastingFilter;

  InputPixelType                        m_LungThreshold;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkMorphologicalOpenningFeatureGenerator.hxx"
#endif

#endif
