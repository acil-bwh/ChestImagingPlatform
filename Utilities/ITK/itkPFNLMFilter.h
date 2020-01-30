/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkPFNLMFilter.h,v $
  Language:  C++
  Date:      $Date: 2006/03/27 17:01:10 $
  Version:   $Revision: 1.15 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkPFNLMFilter_h
#define __itkPFNLMFilter_h

#include "itkImageToImageFilter.h"
#include "itkLSDerivatives.h"
#include "itkImage.h"

namespace itk
{

/** \class PFNLMFilter
 *
 * DO NOT assume a particular image or pixel type, which is, the input image
 * may be a VectorImage as well as an Image obeject with vectorial pixel type.
 *
 * \sa Image
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT PFNLMFilter : public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
	/** Standard class typedefs. */
	typedef PFNLMFilter                             Self;
	/** Convenient typedefs for simplifying declarations. */
	typedef TInputImage                           InputImageType;
	typedef typename InputImageType::Pointer      InputImagePointer;
	typedef typename InputImageType::ConstPointer InputImageConstPointer;
	typedef TOutputImage                          OutputImageType;
	typedef typename OutputImageType::Pointer     OutputImagePointer;
	
	/** Standard class typedefs. */
	typedef ImageToImageFilter< InputImageType, OutputImageType> Superclass;
	typedef SmartPointer<Self>                                   Pointer;
	typedef SmartPointer<const Self>                             ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Run-time type information (and related methods). */
	itkTypeMacro( PFNLMFilter, ImageToImageFilter );
  
	/** Image typedef support. */
	typedef typename InputImageType::PixelType           InputPixelType;
	typedef typename OutputImageType::PixelType          OutputPixelType;
	typedef typename InputImageType::RegionType          InputImageRegionType;
	typedef typename InputImageType::SizeType            InputImageSizeType;
	typedef typename InputImageType::IndexType           InputImageIndexType;
	typedef typename OutputImageType::RegionType         OutputImageRegionType;
	
	
	typedef itk::Image< LSGradientsL2, TInputImage::ImageDimension > FeaturesMapType;
	typedef typename FeaturesMapType::Pointer                        FeaturesMapPointer;
	typedef itk::LSDerivativesL0<InputImageType>                     L0Type;
	typedef typename L0Type::Pointer                                 L0Pointer;
	typedef itk::LSDerivativesL1<TInputImage::ImageDimension>        L1Type;
	typedef typename L1Type::Pointer                                 L1Pointer;
	typedef itk::LSDerivativesL2<TInputImage::ImageDimension>        L2Type;
	typedef typename L2Type::Pointer                                 L2Pointer;
	
	itkSetMacro( Sigma,      float              );
	itkGetMacro( Sigma,      float              );
	itkSetMacro( H,          float              );
	itkGetMacro( H,          float              );
	itkSetMacro( PSTh,       float              );
        itkGetMacro( PSTh,       float              );
	itkSetMacro( RSearch,    InputImageSizeType );
	itkGetMacro( RSearch,    InputImageSizeType );
	itkSetMacro( RComp,      InputImageSizeType );
	itkGetMacro( RComp,      InputImageSizeType );
	
protected:
	PFNLMFilter();
	virtual ~PFNLMFilter();
	// Threaded filter!
#if ITK_VERSION_MAJOR < 4
    void ThreadedGenerateData( const OutputImageRegionType & outputRegionForThread, int threadId );
    
#else
    void ThreadedGenerateData( const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId ) override;
    
#endif
	void GenerateInputRequestedRegion() override;
	void BeforeThreadedGenerateData( void ) override;
	void PrintSelf( std::ostream &os, Indent indent ) const override;
private:
	PFNLMFilter(const Self&);         // purposely not implemented
	void operator=(const Self&);    // purposely not implemented
	float ComputeTraceMO0( const InputImageSizeType& rcomp );
	float ComputeTraceMO1( const InputImageSizeType& rcomp );
	// The standard deviation of noise (in the complex domain)
	float                m_Sigma;
	// The true parameteres of NLM:
	float                m_H;
	float                m_PSTh;
	InputImageSizeType   m_RSearch;
	InputImageSizeType   m_RComp;
	FeaturesMapPointer   m_Features;
};


 
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPFNLMFilter.txx"
#endif

#endif
