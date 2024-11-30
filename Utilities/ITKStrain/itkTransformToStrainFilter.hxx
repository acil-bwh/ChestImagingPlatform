/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkTransformToStrainFilter_hxx
#define itkTransformToStrainFilter_hxx

#include "itkImageRegionIteratorWithIndex.h"
#include "itkTransformToStrainFilter.h"

namespace itk
{

template < typename TTransform, typename TOperatorValue,
           typename TOutputValue >
TransformToStrainFilter< TTransform, TOperatorValue, TOutputValue >
::TransformToStrainFilter():
  m_StrainForm( INFINITESIMAL )
{
  m_DeformationTensor = false;
}

template < typename TTransform, typename TOperatorValue,
           typename TOutputValue >
void
TransformToStrainFilter< TTransform, TOperatorValue, TOutputValue >
::BeforeThreadedGenerateData()
{
  OutputImageType * output = this->GetOutput();
  output->FillBuffer( NumericTraits< OutputPixelType >::Zero );

  const TransformType * input = this->GetTransform();
  if( input == ITK_NULLPTR )
    {
    itkExceptionMacro( "Input transform not available!" );
    }

  const StrainFormType strainForm = this->GetStrainForm();
  if( strainForm != INFINITESIMAL && strainForm != GREENLAGRANGIAN && strainForm != EULERIANALMANSI )
    {
    itkExceptionMacro( "Invalid StrainForm!" );
    }
}

template < typename TTransform, typename TOperatorValue,
           typename TOutputValue >
void
TransformToStrainFilter< TTransform, TOperatorValue, TOutputValue >
::ThreadedGenerateData( const OutputRegionType& region,
                        ThreadIdType itkNotUsed( threadId ) )
{
  const TransformType * input = this->GetTransform();

  OutputImageType * output = this->GetOutput();
  typedef ImageRegionIteratorWithIndex< OutputImageType > ImageIteratorType;
  ImageIteratorType outputIt( output, region );

  typename TransformType::JacobianType identity;
  identity.SetSize( ImageDimension, ImageDimension );
  identity.Fill( 0.0 );
  for( unsigned int i = 0; i < ImageDimension; ++i )
    {
    identity.SetElement( i, i, 1.0);
    }

  // e_ij += 1/2( du_i/dx_j + du_j/dx_i )
  for( outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt )
    {
    const typename OutputImageType::IndexType index = outputIt.GetIndex();
    typename OutputImageType::PointType point;
    output->TransformIndexToPhysicalPoint( index, point );
    typename TransformType::JacobianType jacobian;
    input->ComputeJacobianWithRespectToPosition( point, jacobian );
    typename OutputImageType::PixelType outputPixel = outputIt.Get();
    for( unsigned int i = 0; i < ImageDimension; ++i )
      {
      for( unsigned int j = 0; j < i; ++j )
        {
        outputPixel( i, j ) += jacobian( i, j ) / static_cast< TOutputValue >( 2 );
        }
      for( unsigned int j = i + 1; j < ImageDimension; ++j )
        {
        outputPixel( i, j ) += jacobian( i, j ) / static_cast< TOutputValue >( 2 );
        }
      outputPixel( i, i ) = jacobian( i, i ) - static_cast< TOutputValue >( 1 );
      }
    outputIt.Set( outputPixel );
    }
  switch( m_StrainForm )
    {
  case INFINITESIMAL:
      if (m_DeformationTensor)
        {
        for( outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt )
        {
          typename OutputImageType::PixelType outputPixel = outputIt.Get();
          for( unsigned int i = 0; i < ImageDimension; ++i )
          {
            for( unsigned int j = 0; j < i ; ++j )
            {
              outputPixel(i,j)=static_cast< TOutputValue >( 2 )*outputPixel( i, j );
            }
            // j == i
            outputPixel(i,i)=static_cast< TOutputValue >( 2 )*outputPixel (i, i ) + static_cast< TOutputValue >( 1 );
          }
          outputIt.Set( outputPixel );
        }
      }
      break;
  // e_ij += 1/2 du_m/du_i du_m/du_j
  case GREENLAGRANGIAN:
    for( outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt )
      {
      const typename OutputImageType::IndexType index = outputIt.GetIndex();
      typename OutputImageType::PointType point;
      output->TransformIndexToPhysicalPoint( index, point );
      typename TransformType::JacobianType jacobian;
      input->ComputeJacobianWithRespectToPosition( point, jacobian );
      jacobian -= identity;
      typename OutputImageType::PixelType outputPixel = outputIt.Get();
      for( unsigned int i = 0; i < ImageDimension; ++i )
        {
        for( unsigned int j = 0; j < ImageDimension; ++j )
          {
          for( unsigned int k = 0; k <= j; ++k )
            {
            outputPixel( j, k ) += jacobian( i, j ) * jacobian( i, k ) / static_cast< TOutputValue >( 2 );
            }
          }
        }
        
      //Adjust result if we want to compute the deformation tensor
      if ( m_DeformationTensor)
        {
          for( unsigned int i = 0; i < ImageDimension; ++i )
          {
            for( unsigned int j = 0; j < i ; ++j )
            {
              outputPixel(i,j)=static_cast< TOutputValue >( 2 )*outputPixel( i, j );
            }
            // j == i
            outputPixel(i,i)=static_cast< TOutputValue >( 2 )*outputPixel (i, i ) + static_cast< TOutputValue >( 1 );
          }

        }
        
      outputIt.Set( outputPixel );
      }
      break;
  // e_ij -= 1/2 du_m/du_i du_m/du_j
  case EULERIANALMANSI:
    for( outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt )
      {
      const typename OutputImageType::IndexType index = outputIt.GetIndex();
      typename OutputImageType::PointType point;
      output->TransformIndexToPhysicalPoint( index, point );
      typename TransformType::JacobianType jacobian;
      input->ComputeJacobianWithRespectToPosition( point, jacobian );
      jacobian -= identity;
      typename OutputImageType::PixelType outputPixel = outputIt.Get();
      for( unsigned int i = 0; i < ImageDimension; ++i )
        {
        for( unsigned int j = 0; j < ImageDimension; ++j )
          {
          for( unsigned int k = 0; k <= j; ++k )
            {
            outputPixel( j, k ) -= jacobian( i, j ) * jacobian( i, k ) / static_cast< TOutputValue >( 2 );
            }
          }
        }
      //Adjust result if we want to compute the deformation tensor
      if ( m_DeformationTensor)
      {
        for( unsigned int i = 0; i < ImageDimension; ++i )
        {
          for( unsigned int j = 0; j < i ; ++j )
          {
            outputPixel(i,j)=static_cast< TOutputValue >( 2 )*outputPixel( i, j );
          }
          // j == i
          outputPixel(i,i)=static_cast< TOutputValue >( 2 )*outputPixel (i, i ) + static_cast< TOutputValue >( 1 );
        }
        
      }
      outputIt.Set( outputPixel );
      }
      break;
  default:
    itkExceptionMacro( << "Unknown strain form." );
    }
}

template < typename TTransform, typename TOperatorValue,
           typename TOutputValue >
void
TransformToStrainFilter< TTransform, TOperatorValue, TOutputValue >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "StrainForm: "
    << static_cast< typename NumericTraits< StrainFormType >::PrintType >( m_StrainForm )
    << std::endl;
}
} // end namespace itk

#endif
