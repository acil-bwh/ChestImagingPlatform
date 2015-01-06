/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkCannyEdgesDistanceAdvectionFieldFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkCannyEdgesDistanceAdvectionFieldFeatureGenerator_h
#define __itkCannyEdgesDistanceAdvectionFieldFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkCastImageFilter.h"
#include "itkCannyEdgeDetectionRecursiveGaussianImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkMultiplyImageFilter.h"

namespace itk
{

/** \class CannyEdgesDistanceAdvectionFieldFeatureGenerator
 * \brief Generates an advection feature field by computing the distance 
 * map to the canny edges in the image and modulating it with the 
 * gradient vectors of the distance map.
 *
 * \par Overview
 * The class generates features that can be used as the advection term for 
 * computing a canny level set. The class takes an input image
 *   
 *    Input -> CastToFloat -> DistanceMap  = ImageA
 *    ImageA -> Gradient = ImageB (of covariant vectors)
 * 
 *   Advection Field = ImageA * ImageB
 *
 * The resulting feature is an image of covariant vectors and is ideally used 
 * as the advection term for a level set segmentation module. The term 
 * advects the level set along the gradient of the distance map, helping it
 * lock onto the edges (which are extracted by the canny filter).
 *
 * There are two parameters to this feature generator.
 * (1) UpperThreshold/LowerThreshold: These set the thresholding values of 
 *     the Canny edge detection. The canny algorithm incorporates a 
 *     hysteresis thresholding which is applied to the gradient magnitude
 *     of the smoothed image to find edges.
 * (2) Variance.  Controls the smoothing paramter of the gaussian filtering
 *     done during Canny edge detection. The first step of canny edge 
 *     detection is to smooth the input with a gaussian filter. Second
 *     derivatives etc are computed on the smoothed image.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT CannyEdgesDistanceAdvectionFieldFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef CannyEdgesDistanceAdvectionFieldFeatureGenerator                Self;
  typedef FeatureGenerator<NDimension>                      Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CannyEdgesDistanceAdvectionFieldFeatureGenerator, FeatureGenerator);

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

  itkSetMacro( Sigma, double );
  itkGetMacro( Sigma, double );
  itkSetMacro( UpperThreshold, double );
  itkGetMacro( UpperThreshold, double );
  itkSetMacro( LowerThreshold, double );
  itkGetMacro( LowerThreshold, double );

protected:
  CannyEdgesDistanceAdvectionFieldFeatureGenerator();
  virtual ~CannyEdgesDistanceAdvectionFieldFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  CannyEdgesDistanceAdvectionFieldFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef CastImageFilter<
    InputImageType, InternalImageType >               CastFilterType;
  typedef typename CastFilterType::Pointer            CastFilterPointer;
  typedef CannyEdgeDetectionRecursiveGaussianImageFilter<
    InternalImageType, InternalImageType >            CannyEdgeFilterType;
  typedef typename CannyEdgeFilterType::Pointer       CannyEdgeFilterPointer;

  typedef SignedMaurerDistanceMapImageFilter<
    InternalImageType, InternalImageType >            DistanceMapFilterType;
  typedef typename DistanceMapFilterType::Pointer     DistanceMapFilterPointer;  

  typedef GradientImageFilter< InternalImageType, 
          InternalPixelType, InternalPixelType >              GradientFilterType;
  typedef typename GradientFilterType::Pointer                GradientFilterPointer;
  typedef typename GradientFilterType::OutputImageType        CovariantVectorImageType;

  typedef typename CovariantVectorImageType::PixelType        OutputPixelType;
  typedef ImageSpatialObject< NDimension, OutputPixelType >   OutputImageSpatialObjectType;
  typedef Image< OutputPixelType, Dimension >                 OutputImageType;

  typedef MultiplyImageFilter< 
    CovariantVectorImageType, InternalImageType,
                     CovariantVectorImageType >       MultiplyFilterType;
  typedef typename MultiplyFilterType::Pointer        MultiplyFilterPointer;
  
  CastFilterPointer                                   m_CastFilter;
  DistanceMapFilterPointer                            m_DistanceMapFilter;
  CannyEdgeFilterPointer                              m_CannyFilter;
  GradientFilterPointer                               m_GradientFilter;
  MultiplyFilterPointer                               m_MultiplyFilter;

  double                                              m_UpperThreshold;
  double                                              m_LowerThreshold;
  double                                              m_Sigma;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkCannyEdgesDistanceAdvectionFieldFeatureGenerator.hxx"
#endif

#endif
