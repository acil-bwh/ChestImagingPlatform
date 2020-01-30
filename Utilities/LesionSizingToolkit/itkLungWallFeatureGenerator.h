/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLungWallFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLungWallFeatureGenerator_h
#define __itkLungWallFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkVotingBinaryHoleFillFloodingImageFilter.h"

namespace itk
{

/** \class LungWallFeatureGenerator
 * \brief Generates a feature image 
 *
 * The typical use of this class would be to generate the feature image needed
 * by a Level Set filter to internally compute its speed image. This
 * transformation is very close to a simply thresholding selection on the input
 * image, but with the advantage of a smooth transition of intensities.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT LungWallFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef LungWallFeatureGenerator         Self;
  typedef FeatureGenerator<NDimension>     Superclass;
  typedef SmartPointer<Self>               Pointer;
  typedef SmartPointer<const Self>         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LungWallFeatureGenerator, FeatureGenerator);

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
  LungWallFeatureGenerator();
  virtual ~LungWallFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData () override;

private:
  LungWallFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef float                                       OutputPixelType;
  typedef Image< OutputPixelType, Dimension >         OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef BinaryThresholdImageFilter<
    InputImageType, InternalImageType >                   ThresholdFilterType;
  typedef typename ThresholdFilterType::Pointer           ThresholdFilterPointer;

  typedef VotingBinaryHoleFillFloodingImageFilter<
    InternalImageType, OutputImageType >                  VotingHoleFillingFilterType;
  typedef typename VotingHoleFillingFilterType::Pointer   VotingHoleFillingFilterPointer;

  ThresholdFilterPointer                m_ThresholdFilter;
  VotingHoleFillingFilterPointer        m_VotingHoleFillingFilter;

  InputPixelType                        m_LungThreshold;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkLungWallFeatureGenerator.hxx"
#endif

#endif
