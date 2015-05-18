/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLSDerivatives.txx,v $
  Language:  C++
  Date:      $Date: 2006/01/11 19:43:31 $
  Version:   $Revision: 1.21 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkLSDerivatives_txx
#define _itkLSDerivatives_txx
#include "itkLSDerivatives.h"

#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "math.h"

namespace itk
{




//=====================================================================================================
template< class TInputImage >
LSDerivativesL0< TInputImage >
::LSDerivativesL0()
{
	m_Radius     = 2;
	m_Coordinate = 0;
}

template< unsigned int ImageDimension >
LSDerivativesL1< ImageDimension >
::LSDerivativesL1()
{
	m_Radius     = 2;
	m_Coordinate = 1;
}

template< unsigned int ImageDimension >
LSDerivativesL2< ImageDimension >
::LSDerivativesL2()
{
	m_Radius     = 2;
	m_Coordinate = 2;
}
//=====================================================================================================	






//=====================================================================================================
template< class TInputImage >
void LSDerivativesL0< TInputImage >
::BeforeThreadedGenerateData( void )
{
	if( m_Coordinate>=TInputImage::ImageDimension )
		itkExceptionMacro( << "Filtering direction exceeds image dimensions" );
}

template< unsigned int ImageDimension >
void LSDerivativesL1< ImageDimension >
::BeforeThreadedGenerateData( void )
{
	if( m_Coordinate>=ImageDimension )
		itkExceptionMacro( << "Filtering direction exceeds image dimensions" );
}

template< unsigned int ImageDimension >
void LSDerivativesL2< ImageDimension >
::BeforeThreadedGenerateData( void )
{
	if( m_Coordinate>=ImageDimension )
		itkExceptionMacro( << "Filtering direction exceeds image dimensions" );
}
//=====================================================================================================



	

//=====================================================================================================	
template< class TInputImage >
void LSDerivativesL0< TInputImage >
::GenerateInputRequestedRegion()
{
	// Call the superclass' implementation of this method
	Superclass::GenerateInputRequestedRegion();
		
	// Get pointers to the input and output
	InputImagePointer  inputPtr  = const_cast< TInputImage * >( this->GetInput() );
	OutputImagePointer outputPtr = this->GetOutput();
		
	if ( !inputPtr || !outputPtr ){return;}
		
	InputSizeType size;
	size.Fill( 0 );
	if( m_Coordinate<TInputImage::ImageDimension )
		size[m_Coordinate] = m_Radius;
		
	// Get a copy of the input requested region (should equal the output
	// requested region)
	InputRegionType inputRequestedRegion = inputPtr->GetRequestedRegion();
	inputRequestedRegion.PadByRadius( size );
		
	// Crop the input requested region at the input's largest possible region
	inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() );
	inputPtr->SetRequestedRegion( inputRequestedRegion );
	return;
}

template< unsigned int ImageDimension >
void LSDerivativesL1< ImageDimension >
::GenerateInputRequestedRegion()
{
	// Call the superclass' implementation of this method
	Superclass::GenerateInputRequestedRegion();
		
	// Get pointers to the input and output
	InputImagePointer  inputPtr  = const_cast< InputImageType* >( this->GetInput() );
	OutputImagePointer outputPtr = this->GetOutput();
		
	if ( !inputPtr || !outputPtr ){return;}
		
	InputSizeType size;
	size.Fill( 0 );
	if( m_Coordinate < ImageDimension )
		size[m_Coordinate] = m_Radius;
		
	// Get a copy of the input requested region (should equal the output
	// requested region)
	InputRegionType inputRequestedRegion = inputPtr->GetRequestedRegion();
	inputRequestedRegion.PadByRadius( size );
		
	// Crop the input requested region at the input's largest possible region
	inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() );
	inputPtr->SetRequestedRegion( inputRequestedRegion );
	return;
}

template< unsigned int ImageDimension >
void LSDerivativesL2< ImageDimension >
::GenerateInputRequestedRegion()
{
	// Call the superclass' implementation of this method
	Superclass::GenerateInputRequestedRegion();
		
	// Get pointers to the input and output
	InputImagePointer  inputPtr  = const_cast< InputImageType* >( this->GetInput() );
	OutputImagePointer outputPtr = this->GetOutput();
		
	if ( !inputPtr || !outputPtr ){return;}
		
	InputSizeType size;
	size.Fill( 0 );
	if( m_Coordinate<ImageDimension )
		size[m_Coordinate] = m_Radius;
		
	// Get a copy of the input requested region (should equal the output
	// requested region)
	InputRegionType inputRequestedRegion = inputPtr->GetRequestedRegion();
	inputRequestedRegion.PadByRadius( size );
		
	// Crop the input requested region at the input's largest possible region
	inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() );
	inputPtr->SetRequestedRegion( inputRequestedRegion );
	return;
}
//=====================================================================================================
	
	


//=====================================================================================================
//=====================================================================================================
//=====================================================================================================
//=====================================================================================================
//=====================================================================================================
//=====================================================================================================	

template< class TInputImage >
#if ITK_VERSION_MAJOR < 4
void LSDerivativesL0< TInputImage >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId )
#else
void LSDerivativesL0< TInputImage >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId )
#endif
{
	// Boundary conditions for this filter; Neumann conditions are fine
	ZeroFluxNeumannBoundaryCondition<InputImageType> nbc;	
	// Iterators:
	ConstNeighborhoodIterator<InputImageType>        bit;  // Iterator for the input image
	ImageRegionIterator<OutputImageType>             it;   // Iterator for the output image
	// Input and output
	InputImageConstPointer   input   =  this->GetInput();
	OutputImagePointer       output  =  this->GetOutput();
	//--------------------------------------------------------------------------------------------------------------------
	InputSizeType size;
	size.Fill( 0 );
	size[m_Coordinate] = m_Radius;
	float* lpf = new float[2*m_Radius+1];
	float* hpf = new float[2*m_Radius+1];
	for( int k=-((int)m_Radius); k<=((int)m_Radius); ++k ){
		lpf[k+m_Radius] = 1.0f;
		hpf[k+m_Radius] = (float)k;
	}
	float* weight = new float[m_Radius];
	float  wsum   = itk::NumericTraits<float>::Zero;
	for( int k=0; k<((int)m_Radius); ++k ){
		weight[k]  = ::exp( -((float)(m_Radius-k)*(float)(m_Radius-k))/2.0f );
		wsum      += 2.0f*weight[k];
	}
	wsum += weight[m_Radius-1];
	wsum  = 1.0f/wsum;
	for( int k=0; k<((int)m_Radius); ++k ){
		lpf[k]            *= ( weight[k] * wsum );
		lpf[2*m_Radius-k] *= ( weight[k] * wsum );
		hpf[k]            *= ( weight[k] * wsum );
		hpf[2*m_Radius-k] *= ( weight[k] * wsum );
	}
	lpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	hpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	delete[] weight;
	//--------------------------------------------------------------------------------------------------------------------
	// Auxiliar values to store the filtered values:
	float           ip;
	OutputPixelType op;
	//--------------------------------------------------------------------------------------------------------------------
	// Find the data-set boundary "faces"
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType           faceList;
	NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>                                  bC;
	
	faceList = bC( input, outputRegionForThread, size );
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType::iterator fit;
	
	for ( fit=faceList.begin(); fit!=faceList.end(); ++fit ){ // Iterate through facets
		bit = ConstNeighborhoodIterator<InputImageType>(     size, input, *fit  );
		it  = ImageRegionIterator<OutputImageType>(          output,      *fit  );
		// Boundary condition:
		bit.OverrideBoundaryCondition(&nbc);
		for( bit.GoToBegin(),it.GoToBegin(); !bit.IsAtEnd(); ++bit,++it ){   // Iterate through pixels in the current facet
			// Auxiliar value to store filtered values:
			op.L = itk::NumericTraits<float>::Zero;
			op.H = itk::NumericTraits<float>::Zero;
			for( unsigned int k=0; k<2*m_Radius+1; ++k ){
				ip     = (float)( bit.GetPixel(k) );
				op.L  += ip * lpf[k];
				op.H  += ip * hpf[k];
			}
			//-------------------------------------------------------------------------------------------------------------
			// Set the output pixel
			it.Set( op );
		}
	}
	delete[] lpf;
	delete[] hpf;
}
//=====================================================================================================
//=====================================================================================================
//=====================================================================================================
template< unsigned int ImageDimension >
#if ITK_VERSION_MAJOR < 4
void LSDerivativesL1< ImageDimension >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId )
#else
void LSDerivativesL1< ImageDimension >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId )
#endif
{
	// Boundary conditions for this filter; Neumann conditions are fine
	ZeroFluxNeumannBoundaryCondition<InputImageType> nbc;	
	// Iterators:
	ConstNeighborhoodIterator<InputImageType>        bit;  // Iterator for the input image
	ImageRegionIterator<OutputImageType>             it;   // Iterator for the output image
	// Input and output
	InputImageConstPointer   input   =  this->GetInput();
	OutputImagePointer       output  =  this->GetOutput();
	//--------------------------------------------------------------------------------------------------------------------
	InputSizeType size;
	size.Fill( 0 );
	size[m_Coordinate] = m_Radius;
	float* lpf = new float[2*m_Radius+1];
	float* hpf = new float[2*m_Radius+1];
	for( int k=-((int)m_Radius); k<=((int)m_Radius); ++k ){
		lpf[k+m_Radius] = 1.0f;
		hpf[k+m_Radius] = (float)k;
	}
	float* weight = new float[m_Radius];
	float  wsum   = itk::NumericTraits<float>::Zero;
	for( int k=0; k<((int)m_Radius); ++k ){
		weight[k]  = ::exp( -((float)(m_Radius-k)*(float)(m_Radius-k))/2.0f );
		wsum      += 2.0f*weight[k];
	}
	wsum += weight[m_Radius-1];
	wsum  = 1.0f/wsum;
	for( int k=0; k<((int)m_Radius); ++k ){
		lpf[k]            *= ( weight[k] * wsum );
		lpf[2*m_Radius-k] *= ( weight[k] * wsum );
		hpf[k]            *= ( weight[k] * wsum );
		hpf[2*m_Radius-k] *= ( weight[k] * wsum );
	}
	lpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	hpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	delete[] weight;
	//--------------------------------------------------------------------------------------------------------------------
	// Auxiliar values to store the filtered values:
	InputPixelType  ip;
	OutputPixelType op;
	//--------------------------------------------------------------------------------------------------------------------
	// Find the data-set boundary "faces"
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType           faceList;
	NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>                                  bC;
	
	faceList = bC( input, outputRegionForThread, size );
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType::iterator fit;
	
	for ( fit=faceList.begin(); fit!=faceList.end(); ++fit ){ // Iterate through facets
		bit = ConstNeighborhoodIterator<InputImageType>(     size, input, *fit  );
		it  = ImageRegionIterator<OutputImageType>(          output,      *fit  );
		// Boundary condition:
		bit.OverrideBoundaryCondition(&nbc);
		for( bit.GoToBegin(),it.GoToBegin(); !bit.IsAtEnd(); ++bit,++it ){   // Iterate through pixels in the current facet
			// Auxiliar value to store filtered values:
			op.LL = itk::NumericTraits<float>::Zero;
			op.HL = itk::NumericTraits<float>::Zero;
			op.LH = itk::NumericTraits<float>::Zero;
			for( unsigned int k=0; k<2*m_Radius+1; ++k ){
				ip     = bit.GetPixel(k);
				op.LL += ( ip.L ) * lpf[k];
				op.HL += ( ip.H ) * lpf[k];
				op.LH += ( ip.L ) * hpf[k];
			}
			//-------------------------------------------------------------------------------------------------------------
			// Set the output pixel
			it.Set( op );
		}
	}
	delete[] lpf;
	delete[] hpf;
}
//=====================================================================================================
//=====================================================================================================
//=====================================================================================================	
template< unsigned int ImageDimension >


#if ITK_VERSION_MAJOR < 4
void LSDerivativesL2< ImageDimension >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, int threadId )
#else
void LSDerivativesL2< ImageDimension >
::ThreadedGenerateData( const OutputRegionType& outputRegionForThread, ThreadIdType threadId )
#endif
{
	// Boundary conditions for this filter; Neumann conditions are fine
	ZeroFluxNeumannBoundaryCondition<InputImageType> nbc;	
	// Iterators:
	ConstNeighborhoodIterator<InputImageType>        bit;  // Iterator for the input image
	ImageRegionIterator<OutputImageType>             it;   // Iterator for the output image
	// Input and output
	InputImageConstPointer   input   =  this->GetInput();
	OutputImagePointer       output  =  this->GetOutput();
	//--------------------------------------------------------------------------------------------------------------------
	InputSizeType size;
	size.Fill( 0 );
	size[m_Coordinate] = m_Radius;
	float* lpf = new float[2*m_Radius+1];
	float* hpf = new float[2*m_Radius+1];
	for( int k=-((int)m_Radius); k<=((int)m_Radius); ++k ){
		lpf[k+m_Radius] = 1.0f;
		hpf[k+m_Radius] = (float)k;
	}
	float* weight = new float[m_Radius];
	float  wsum   = itk::NumericTraits<float>::Zero;
	for( int k=0; k<((int)m_Radius); ++k ){
		weight[k]  = ::exp( -((float)(m_Radius-k)*(float)(m_Radius-k))/2.0f );
		wsum      += 2.0f*weight[k];
	}
	wsum += weight[m_Radius-1];
	wsum  = 1.0f/wsum;
	for( int k=0; k<((int)m_Radius); ++k ){
		lpf[k]            *= ( weight[k] * wsum );
		lpf[2*m_Radius-k] *= ( weight[k] * wsum );
		hpf[k]            *= ( weight[k] * wsum );
		hpf[2*m_Radius-k] *= ( weight[k] * wsum );
	}
	lpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	hpf[m_Radius] *= ( weight[m_Radius-1] * wsum );
	delete[] weight;
	//--------------------------------------------------------------------------------------------------------------------
	// Auxiliar values to store the filtered values:
	InputPixelType  ip;
	OutputPixelType op;
	//--------------------------------------------------------------------------------------------------------------------
	// Find the data-set boundary "faces"
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType           faceList;
	NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>                                  bC;
	
	faceList = bC( input, outputRegionForThread, size );
	typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType::iterator fit;
	
	for ( fit=faceList.begin(); fit!=faceList.end(); ++fit ){ // Iterate through facets
		bit = ConstNeighborhoodIterator<InputImageType>(     size, input, *fit  );
		it  = ImageRegionIterator<OutputImageType>(          output,      *fit  );
		// Boundary condition:
		bit.OverrideBoundaryCondition(&nbc);
		for( bit.GoToBegin(),it.GoToBegin(); !bit.IsAtEnd(); ++bit,++it ){   // Iterate through pixels in the current facet
			// Auxiliar value to store filtered values:
			op.LLL = itk::NumericTraits<float>::Zero;
			op.HLL = itk::NumericTraits<float>::Zero;
			op.LHL = itk::NumericTraits<float>::Zero;
			op.LLH = itk::NumericTraits<float>::Zero;
			for( unsigned int k=0; k<2*m_Radius+1; ++k ){
				ip      = bit.GetPixel(k);
				op.LLL += ( ip.LL ) * lpf[k];
				op.HLL += ( ip.HL ) * lpf[k];
				op.LHL += ( ip.LH ) * lpf[k];
				op.LLH += ( ip.LL ) * hpf[k];
			}
			//-------------------------------------------------------------------------------------------------------------
			// Set the output pixel
			it.Set( op );
		}
	}
	delete[] lpf;
	delete[] hpf;
}
	

	
} // end namespace itk


#endif
