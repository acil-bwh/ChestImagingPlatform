/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSatoLocalStructureFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSatoLocalStructureFeatureGenerator_h
#define __itkSatoLocalStructureFeatureGenerator_h

#include "itkFeatureGenerator.h"
#include "itkImage.h"
#include "itkImageSpatialObject.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkLocalStructureImageFilter.h"

namespace itk
{

/** \class SatoLocalStructureFeatureGenerator
 * \brief Generates a feature image by computing measures based on the Hessian Eigenvalues.
 *
 * The typical use of this class would be to generate a map of {blobs, tubes, sheets}.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT SatoLocalStructureFeatureGenerator : public FeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef SatoLocalStructureFeatureGenerator                Self;
  typedef FeatureGenerator<NDimension>                      Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SatoLocalStructureFeatureGenerator, FeatureGenerator);

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

  /** Alpha value to be used in the Sato Vesselness filter. */
  itkSetMacro( Alpha, double );
  itkGetMacro( Alpha, double );

  /** Gamma value to be used in the Sato Vesselness filter. */
  itkSetMacro( Gamma, double );
  itkGetMacro( Gamma, double );

protected:
  SatoLocalStructureFeatureGenerator();
  virtual ~SatoLocalStructureFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData ();

private:
  SatoLocalStructureFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef InternalPixelType                           OutputPixelType;
  typedef InternalImageType                           OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef HessianRecursiveGaussianImageFilter< InputImageType >     HessianFilterType;
  typedef typename HessianFilterType::OutputImageType               HessianImageType;
  typedef typename HessianImageType::PixelType                      HessianPixelType;

  typedef  FixedArray< double, HessianPixelType::Dimension >   EigenValueArrayType;
  typedef  Image< EigenValueArrayType, Dimension >             EigenValueImageType;

  typedef  SymmetricEigenAnalysisImageFilter< HessianImageType, EigenValueImageType >     EigenAnalysisFilterType;

  typedef  LocalStructureImageFilter< EigenValueImageType, OutputImageType >     LocalStructureFilterType;

  typename HessianFilterType::Pointer             m_HessianFilter;
  typename EigenAnalysisFilterType::Pointer       m_EigenAnalysisFilter;
  typename LocalStructureFilterType::Pointer      m_LocalStructureFilter;

  double      m_Sigma;
  double      m_Alpha;
  double      m_Gamma;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkSatoLocalStructureFeatureGenerator.hxx"
#endif

#endif
