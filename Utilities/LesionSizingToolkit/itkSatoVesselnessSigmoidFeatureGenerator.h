/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkSatoVesselnessSigmoidFeatureGenerator.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSatoVesselnessSigmoidFeatureGenerator_h
#define __itkSatoVesselnessSigmoidFeatureGenerator_h

#include "itkSatoVesselnessFeatureGenerator.h"
#include "itkSigmoidImageFilter.h"

namespace itk
{

/** \class SatoVesselnessSigmoidFeatureGenerator
 * \brief Generates a feature image by computing the Sato Vesselness measure of the
 * input image and applying a sigmoid transformation to it.
 *
 * The typical use of this class would be to generate the Vessel-map needed by a
 * Level Set filter to internally compute its speed image.
 *
 * SpatialObjects are used as inputs and outputs of this class.
 *
 * \ingroup SpatialObjectFilters
 * \ingroup LesionSizingToolkit
 */
template <unsigned int NDimension>
class ITK_EXPORT SatoVesselnessSigmoidFeatureGenerator : public SatoVesselnessFeatureGenerator<NDimension>
{
public:
  /** Standard class typedefs. */
  typedef SatoVesselnessSigmoidFeatureGenerator         Self;
  typedef SatoVesselnessFeatureGenerator<NDimension>    Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SatoVesselnessSigmoidFeatureGenerator, SatoVesselnessFeatureGenerator);

  /** Dimension of the space */
  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Alpha value to be used in the Sigmoid transformation. It should be a
   * negative number if the feature is going to be used to exclude vessels, and
   * it should be a positive number if the feature is intended to include
   * vessels. */
  itkSetMacro( SigmoidAlpha, double );
  itkGetMacro( SigmoidAlpha, double );

  /** Beta value to be used in the Sigmoid transformation. It should be close
   * to the Vesselness value that is considered to be a threshold in Vesselness. */
  itkSetMacro( SigmoidBeta, double );
  itkGetMacro( SigmoidBeta, double );

protected:
  SatoVesselnessSigmoidFeatureGenerator();
  virtual ~SatoVesselnessSigmoidFeatureGenerator();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the segmentation. */
  void  GenerateData () override;

private:
  SatoVesselnessSigmoidFeatureGenerator(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef float                                       InternalPixelType;
  typedef Image< InternalPixelType, Dimension >       InternalImageType;

  typedef InternalPixelType                           OutputPixelType;
  typedef InternalImageType                           OutputImageType;

  typedef ImageSpatialObject< NDimension, OutputPixelType >  OutputImageSpatialObjectType;

  typedef SigmoidImageFilter< InternalImageType, InternalImageType > SigmoidFilterType;

  typename SigmoidFilterType::Pointer                 m_SigmoidFilter;

  double      m_SigmoidAlpha;
  double      m_SigmoidBeta;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
# include "itkSatoVesselnessSigmoidFeatureGenerator.hxx"
#endif

#endif
