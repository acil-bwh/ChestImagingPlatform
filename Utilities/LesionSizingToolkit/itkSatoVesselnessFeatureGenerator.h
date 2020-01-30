/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSatoVesselnessFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSatoVesselnessFeatureGenerator_h
#define __itkSatoVesselnessFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkHessian3DToVesselnessMeasureImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkVesselEnhancingDiffusion3DImageFilter.h"

namespace itk
{

/** \class SatoVesselnessFeatureGenerator
 * \brief Generates a feature image by computing the Sato Vesselness measure of
 * the input image. 
 *
 * The typical use of this class would be to generate the Vesselness-map needed
 * by a Level Set filter to internally compute its speed image.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT SatoVesselnessFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef SatoVesselnessFeatureGenerator          Self;
  typedef FeatureGenerator<NDimension>            Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SatoVesselnessFeatureGenerator, FeatureGenerator);

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

  /** Sigma value to be used in the Gaussian smoothing preceding the
   * Hessian computation. */
  itkSetMacro( Sigma, double );
  itkGetMacro( Sigma, double );

  /** Alpha1 value to be used in the Sato Vesselness filter. */
  itkSetMacro( Alpha1, double );
  itkGetMacro( Alpha1, double );

  /** Alpha2 value to be used in the Sato Vesselness filter. */
  itkSetMacro( Alpha2, double );
  itkGetMacro( Alpha2, double );

  /** Use vessel enhancing diffusion ? Defaults to false. */
  itkSetMacro( UseVesselEnhancingDiffusion, bool );
  itkGetMacro( UseVesselEnhancingDiffusion, bool );
  itkBooleanMacro( UseVesselEnhancingDiffusion );

protected:
  SatoVesselnessFeatureGenerator();
  virtual ~SatoVesselnessFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData () override;

private:
  SatoVesselnessFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef InternalPixelType                           OutputPixelType;
  typedef InternalImageType                           OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef HessianRecursiveGaussianImageFilter< InputImageType >         HessianFilterType;
  typedef Hessian3DToVesselnessMeasureImageFilter< InternalPixelType >  VesselnessMeasureFilterType;
  typedef VesselEnhancingDiffusion3DImageFilter< InputPixelType, Dimension > VesselEnhancingDiffusionFilterType;

  typename HessianFilterType::Pointer                     m_HessianFilter;
  typename VesselnessMeasureFilterType::Pointer           m_VesselnessFilter;
  typename VesselEnhancingDiffusionFilterType::Pointer    m_VesselEnhancingDiffusionFilter;

  double      m_Sigma;
  double      m_Alpha1;
  double      m_Alpha2;
  bool        m_UseVesselEnhancingDiffusion;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkSatoVesselnessFeatureGenerator.hxx"
#endif

#endif
