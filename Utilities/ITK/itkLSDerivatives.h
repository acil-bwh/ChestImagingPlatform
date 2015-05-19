/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLSDerivatives.h,v $
  Language:  C++
  Date:      $Date: 2006/03/27 17:01:10 $
  Version:   $Revision: 1.15 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLSDerivatives_h
#define __itkLSDerivatives_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

namespace itk
{
	
	typedef struct LSGradientsL0{
		float L;
		float H;
	} LSGradientsL0;
	
	typedef struct LSGradientsL1{
		float LL;
		float HL;
		float LH;
	} LSGradientsL1;
	
	typedef struct LSGradientsL2{
		float LLL;
		float HLL;
		float LHL;
		float LLH;
	} LSGradientsL2;


/** \class LSDerivativesCalculator */
template <class TInputImage>
class ITK_EXPORT LSDerivativesL0 : public ImageToImageFilter< TInputImage, itk::Image<LSGradientsL0,TInputImage::ImageDimension> >
{
public:
	/** Standard class typedefs. */
	typedef LSDerivativesL0                       Self;
	/** Convenient typedefs for simplifying declarations. */
	typedef TInputImage                           InputImageType;
	typedef typename InputImageType::Pointer      InputImagePointer;
	typedef typename InputImageType::ConstPointer InputImageConstPointer;
	typedef typename itk::Image<LSGradientsL0,TInputImage::ImageDimension >
	                                              OutputImageType;
	typedef typename OutputImageType::Pointer     OutputImagePointer;
	
	/** Standard class typedefs. */
	typedef ImageToImageFilter< InputImageType, OutputImageType> Superclass;
	typedef SmartPointer<Self>                                   Pointer;
	typedef SmartPointer<const Self>                             ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Run-time type information (and related methods). */
	itkTypeMacro( LSDerivativesL0, ImageToImageFilter );
  
	/** Image typedef support. */
	typedef typename InputImageType::PixelType           InputPixelType;
	typedef typename OutputImageType::PixelType          OutputPixelType;
	typedef typename InputImageType::RegionType          InputRegionType;
	typedef typename InputImageType::SizeType            InputSizeType;
	typedef typename InputImageType::IndexType           InputIndexType;
	typedef typename OutputImageType::RegionType         OutputRegionType;

  
	/** Set and get the parameters */
	itkSetMacro( Radius,     unsigned int );
	itkGetMacro( Radius,     unsigned int );
	itkSetMacro( Coordinate, unsigned int );
	itkGetMacro( Coordinate, unsigned int );
protected:
	LSDerivativesL0();
	virtual ~LSDerivativesL0() {}
	// Threaded filter!
#if ITK_VERSION_MAJOR < 4
    void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId );
    
#else
    void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId );
    
#endif
	void BeforeThreadedGenerateData();
	void GenerateInputRequestedRegion();
private:
	LSDerivativesL0(const Self&);   // purposely not implemented
	void operator=(const Self&);    // purposely not implemented
	unsigned int m_Radius;
	unsigned int m_Coordinate;
};
	
	
	
	template <unsigned int ImageDimension>
	class ITK_EXPORT LSDerivativesL1 : public ImageToImageFilter< itk::Image<LSGradientsL0,ImageDimension>, itk::Image<LSGradientsL1,ImageDimension> >
	{
	public:
		/** Standard class typedefs. */
		typedef LSDerivativesL1                             Self;
		/** Convenient typedefs for simplifying declarations. */
		typedef itk::Image<LSGradientsL0,ImageDimension >   InputImageType;
		typedef typename InputImageType::Pointer            InputImagePointer;
		typedef typename InputImageType::ConstPointer       InputImageConstPointer;
		typedef itk::Image<LSGradientsL1,ImageDimension >   OutputImageType;
		typedef typename OutputImageType::Pointer           OutputImagePointer;
		
		/** Standard class typedefs. */
		typedef ImageToImageFilter< InputImageType, OutputImageType> Superclass;
		typedef SmartPointer<Self>                                   Pointer;
		typedef SmartPointer<const Self>                             ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( LSDerivativesL1, ImageToImageFilter );
		
		/** Image typedef support. */
		typedef typename InputImageType::PixelType           InputPixelType;
		typedef typename OutputImageType::PixelType          OutputPixelType;
		typedef typename InputImageType::RegionType          InputRegionType;
		typedef typename InputImageType::SizeType            InputSizeType;
		typedef typename InputImageType::IndexType           InputIndexType;
		typedef typename OutputImageType::RegionType         OutputRegionType;
		
		
		/** Set and get the parameters */
		itkSetMacro( Radius,     unsigned int );
		itkGetMacro( Radius,     unsigned int );
		itkSetMacro( Coordinate, unsigned int );
		itkGetMacro( Coordinate, unsigned int );
	protected:
		LSDerivativesL1();
		virtual ~LSDerivativesL1() {}
		// Threaded filter!
#if ITK_VERSION_MAJOR < 4
        void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId );
        
#else
        void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId );
        
#endif
		void BeforeThreadedGenerateData();
		void GenerateInputRequestedRegion();
	private:
		LSDerivativesL1(const Self&);   // purposely not implemented
		void operator=(const Self&);    // purposely not implemented
		unsigned int m_Radius;
		unsigned int m_Coordinate;
	};
	
	
	
	template <unsigned int ImageDimension>
	class ITK_EXPORT LSDerivativesL2 : public ImageToImageFilter< itk::Image<LSGradientsL1,ImageDimension>, itk::Image<LSGradientsL2,ImageDimension> >
	{
	public:
		/** Standard class typedefs. */
		typedef LSDerivativesL2                             Self;
		/** Convenient typedefs for simplifying declarations. */
		typedef itk::Image<LSGradientsL1,ImageDimension >   InputImageType;
		typedef typename InputImageType::Pointer            InputImagePointer;
		typedef typename InputImageType::ConstPointer       InputImageConstPointer;
		typedef itk::Image<LSGradientsL2,ImageDimension >   OutputImageType;
		typedef typename OutputImageType::Pointer           OutputImagePointer;
		
		/** Standard class typedefs. */
		typedef ImageToImageFilter< InputImageType, OutputImageType> Superclass;
		typedef SmartPointer<Self>                                   Pointer;
		typedef SmartPointer<const Self>                             ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( LSDerivativesL2, ImageToImageFilter );
		
		/** Image typedef support. */
		typedef typename InputImageType::PixelType           InputPixelType;
		typedef typename OutputImageType::PixelType          OutputPixelType;
		typedef typename InputImageType::RegionType          InputRegionType;
		typedef typename InputImageType::SizeType            InputSizeType;
		typedef typename InputImageType::IndexType           InputIndexType;
		typedef typename OutputImageType::RegionType         OutputRegionType;
		
		
		/** Set and get the parameters */
		itkSetMacro( Radius,     unsigned int );
		itkGetMacro( Radius,     unsigned int );
		itkSetMacro( Coordinate, unsigned int );
		itkGetMacro( Coordinate, unsigned int );
	protected:
		LSDerivativesL2();
		virtual ~LSDerivativesL2() {}
		// Threaded filter!
#if ITK_VERSION_MAJOR < 4
        void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId );
        
#else
        void ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId );
        
#endif
		void BeforeThreadedGenerateData();
		void GenerateInputRequestedRegion();
	private:
		LSDerivativesL2(const Self&);   // purposely not implemented
		void operator=(const Self&);    // purposely not implemented
		unsigned int m_Radius;
		unsigned int m_Coordinate;
	};


 
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLSDerivatives.txx"
#endif

#endif
