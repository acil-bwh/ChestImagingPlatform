/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkIsotropicResamplerImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkIsotropicResamplerImageFilter_h
#define __itkIsotropicResamplerImageFilter_h

#include "itkResampleImageFilter.h"
#include "itkImage.h"
#include "itkBSplineInterpolateImageFunction.h"

namespace itk
{

/** \class IsotropicResamplerImageFilter
 *
 * \brief Resamples the image to an isotropic resolution.
 *
 * This class resamples an image using BSplineInterpolator and produces
 * an isotropic image.
 *
 *\ingroup LesionSizingToolkit
 * \ingroup LesionSizingToolkit
 */
template<class TInputImage, class TOutputImage>
class IsotropicResamplerImageFilter
  : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard "Self" & Superclass typedef.  */
  typedef IsotropicResamplerImageFilter                 Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  /** Image typedef support   */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;
  typedef typename OutputImageType::Pointer   OutputImagePointer;
      
  /** SmartPointer typedef support  */
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Define pixel types. */
  typedef typename TOutputImage::PixelType        OutputImagePixelType;
  typedef typename InputImageType::SizeType       SizeType;
  typedef typename SizeType::SizeValueType        SizeValueType;
  typedef typename InputImageType::SpacingType    SpacingType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IsotropicResamplerImageFilter, ImageToImageFilter);

  /** ImageDimension constant    */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  itkSetMacro( OutputSpacing, SpacingType );
  itkGetMacro( OutputSpacing, SpacingType );

  itkSetMacro( DefaultPixelValue, OutputImagePixelType );
  itkGetMacro( DefaultPixelValue, OutputImagePixelType );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  /** End concept checking */
#endif

  /** Override the superclass implementation so as to set the flag on all the
   * filters within our lesion segmentation pipeline */
  virtual void SetAbortGenerateData( const bool ) override;

  /** ResampleImageFilter produces an image which is a different size
   * than its input.  As such, it needs to provide an implementation
   * for GenerateOutputInformation() in order to inform the pipeline
   * execution model.  The original documentation of this method is
   * below. \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation( void ) override;

protected:
  IsotropicResamplerImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const override;

  void GenerateData() override;

private:
  virtual ~IsotropicResamplerImageFilter();
  IsotropicResamplerImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  SpacingType m_OutputSpacing;
  typedef ResampleImageFilter< TInputImage, TOutputImage > ResampleFilterType;
  typedef typename ResampleFilterType::Pointer             ResampleFilterPointer;

  ResampleFilterPointer     m_ResampleFilter;
  OutputImagePixelType      m_DefaultPixelValue;
};

} //end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIsotropicResamplerImageFilter.hxx"
#endif

#endif
