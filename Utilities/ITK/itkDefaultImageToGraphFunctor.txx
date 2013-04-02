/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkDefaultImageToGraphFunctor.txx,v $
  Language:  C++
  Date:      $Date: 2009/02/09 21:38:19 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDefaultImageToGraphFunctor_txx
#define __itkDefaultImageToGraphFunctor_txx

#include "itkDefaultImageToGraphFunctor.h"

namespace itk
{

/**
 *
 */
template <typename TInputImage, typename TOutputGraph>
ImageToGraphFunctor<TInputImage, TOutputGraph>
::ImageToGraphFunctor()
{
  // Modify superclass default values, can be overwritten by subclasses
  this->SetNumberOfRequiredInputs( 1 );
  this->m_BackgroundValue = static_cast<PixelType>( 0 );
  this->m_ExcludeBackground = false;
  this->m_StrideTable.SetSize( ImageDimension );
  this->SetRadius( 1 );
}

/**
 *
 */
template <typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::SetInput( const InputImageType *input )
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput( 0,
    const_cast< InputImageType * >( input ) );
}

/**
 * Connect one of the operands for pixel-wise addition
 */
template <typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::SetInput( unsigned int index, const TInputImage *image )
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput( index,
    const_cast< TInputImage *>( image ) );
}

/**
 *
 */
template <typename TInputImage, typename TOutputGraph>
const typename ImageToGraphFunctor<TInputImage, TOutputGraph>::InputImageType *
ImageToGraphFunctor<TInputImage, TOutputGraph>
::GetInput( void )
{
  if( this->GetNumberOfInputs() < 1 )
    {
    return 0;
    }

  return static_cast<const TInputImage *>
    ( this->ProcessObject::GetInput( 0 ) );
}

/**
 *
 */
template <typename TInputImage, typename TOutputGraph>
const typename ImageToGraphFunctor<TInputImage, TOutputGraph>::InputImageType *
ImageToGraphFunctor<TInputImage, TOutputGraph>
::GetInput( unsigned int idx )
{
  return static_cast<const TInputImage *>
    ( this->ProcessObject::GetInput( idx ) );
}

template<typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::ActivateIndex( const unsigned int n )
{
  // Insert so that the list remains ordered.
  typename IndexListType::iterator it = this->m_ActiveIndexList.begin();

  if( this->m_ActiveIndexList.empty() )
    {
    this->m_ActiveIndexList.push_front( n );
    }
  else
    {
    while( n > *it )
      {
      it++;
      if( it == this->m_ActiveIndexList.end() )
        {
        break;
        }
      }
    if( it == this->m_ActiveIndexList.end() )
      {
      this->m_ActiveIndexList.insert( it, n );
      }
    else if( n != *it )
      {
      this->m_ActiveIndexList.insert( it, n );
      }
    }
}

template<typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::DeactivateIndex( const unsigned int n )
{
  typename IndexListType::iterator it = this->m_ActiveIndexList.begin();

  if( this->m_ActiveIndexList.empty() )
    {
    return;
    }
  else
    {
    while(n != *it)
      {
      it++;
      if(it == this->m_ActiveIndexList.end())
        {
        return;
        }
      }
    this->m_ActiveIndexList.erase(it);
    }
}


template<typename TInputImage, typename TOutputGraph>
unsigned int
ImageToGraphFunctor<TInputImage, TOutputGraph>
::GetNeighborhoodIndex(const OffsetType &o) const
{
  unsigned int idx = static_cast<unsigned int>( 
    0.5 * this->m_NumberOfPixelsInNeighborhood );
  for( unsigned i = 0; i < ImageDimension; ++i )
    {
    idx += ( o[i] * static_cast<long>( this->m_StrideTable[i] ) );
    }
  return idx;
}

template<typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::ComputeNeighborhoodStrideTable()
{
  unsigned int stride;
  unsigned int accum;

  for( unsigned int dim = 0; dim < ImageDimension; ++dim )
    {
    stride = 0;
    accum = 1;

    for( unsigned int i = 0; i < ImageDimension; ++i )
      {
      if( i == dim )
        {
        stride = accum;
        }
      accum *= ( 2*this->m_Radius[i]+1 );
      }
    this->m_StrideTable[dim] = stride;
    }
}

template<typename TInputImage, typename TOutputGraph>
void
ImageToGraphFunctor<TInputImage, TOutputGraph>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Exclude Background = "
     << (this->m_ExcludeBackground ? "True" : "False") << std::endl;
  if( this->m_ExcludeBackground )
    {
    os << indent << "Background Value = "
       << this->m_BackgroundValue << std::endl;
    }
  os << indent << "Radius = " << this->m_Radius << std::endl;
  os << indent << "Number of pixels in neighborhood = "
     << this->m_NumberOfPixelsInNeighborhood << std::endl;
}

} // end namespace itk

#endif
