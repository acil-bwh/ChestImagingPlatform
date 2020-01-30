/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkCannyEdgesFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCannyEdgesFeatureGenerator_h
#define __itkCannyEdgesFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkCannyEdgeDetectionRecursiveGaussianImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"

namespace itk
{

/** \class CannyEdgesFeatureGenerator
 * \brief Generates a feature image by computing the inverse map to the canny
 * edges in the image.
 *
 * \par Overview
 * The class generates features that can be used as the speed term for 
 * computing a canny level set. The class takes an input image
 *   
 *    Input -> CastToFloat -> CannyEdgeFilter -> InvertFilter
 *
 * The resulting feature is ideally used as the speed term for a level set
 * segmentation module. The speed feature generated is designed to lock
 * onto edges (which are extracted by the canny filter).
 *
 * There are two parameters to this feature generator.
 * (1) UpperThreshold/LowerThreshold: These set the thresholding values of 
 *     the Canny edge detection. The canny algorithm incorporates a 
 *     hysteresis thresholding which is applied to the gradient magnitude
 *     of the smoothed image to find edges.
 * (2) Sigma.  Controls the smoothing parameter of the Gaussian filtering
 *     done during Canny edge detection. The first step of canny edge 
 *     detection is to smooth the input with a Gaussian filter. Second
 *     derivatives etc are computed on the smoothed image.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT CannyEdgesFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef CannyEdgesFeatureGenerator                        Self;
  typedef FeatureGenerator<NDimension>                      Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CannyEdgesFeatureGenerator, FeatureGenerator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Type of spatialObject that will be passed as input to this
   * feature generator. */
  typedef signed short                                      InputPixelType;
  typedef Image< InputPixelType, Dimension >                InputImageType;
  typedef ImageSpatialObject< NDimension, InputPixelType >  InputImageSpatialObjectType;
  typedef typename InputImageSpatialObjectType::Pointer     InputImageSpatialObjectPointer;
  typedef typename Superclass::SpatialObjectType            SpatialObjectType;

  typedef typename NumericTraits<InputPixelType>::ScalarRealType ScalarRealType;
  typedef FixedArray< ScalarRealType, itkGetStaticConstMacro(Dimension) > SigmaArrayType;
  
  /** Input data that will be used for generating the feature. */
  using ProcessObject::SetInput;
  void SetInput( const SpatialObjectType * input );
  const SpatialObjectType * GetInput() const;

  /** Output data that carries the feature in the form of a
   * SpatialObject. */
  const SpatialObjectType * GetFeature() const;

  /** Set Sigma value. Sigma is measured in the units of image spacing. You 
    may use the method SetSigma to set the same value across each axis or
    use the method SetSigmaArray if you need different values along each
    axis. */
  void SetSigmaArray( const SigmaArrayType & sigmas );
  void SetSigma( ScalarRealType sigma );
  SigmaArrayType GetSigmaArray() const;
  ScalarRealType GetSigma() const;
  
  itkSetMacro( UpperThreshold, double );
  itkGetMacro( UpperThreshold, double );
  itkSetMacro( LowerThreshold, double );
  itkGetMacro( LowerThreshold, double );

protected:
  CannyEdgesFeatureGenerator();
  virtual ~CannyEdgesFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData () override;

private:
  CannyEdgesFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef InternalPixelType                           OutputPixelType;
  typedef InternalImageType                           OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef CastImageFilter<
    InputImageType, InternalImageType >               CastFilterType;
  typedef typename CastFilterType::Pointer            CastFilterPointer;
  typedef CannyEdgeDetectionRecursiveGaussianImageFilter<
    InternalImageType, InternalImageType >            CannyEdgeFilterType;
  typedef typename CannyEdgeFilterType::Pointer       CannyEdgeFilterPointer;

  typedef IntensityWindowingImageFilter<
    InternalImageType, InternalImageType >            RescaleFilterType;
  typedef typename RescaleFilterType::Pointer         RescaleFilterPointer;

  CastFilterPointer                   m_CastFilter;
  RescaleFilterPointer                m_RescaleFilter;
  CannyEdgeFilterPointer              m_CannyFilter;

  double                              m_UpperThreshold;
  double                              m_LowerThreshold;

  /** Standard deviation of the gaussian used for smoothing */
  SigmaArrayType                      m_Sigma;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkCannyEdgesFeatureGenerator.hxx"
#endif

#endif
