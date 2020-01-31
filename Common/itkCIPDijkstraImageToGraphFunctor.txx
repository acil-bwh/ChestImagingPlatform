/**
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 */
#ifndef __itkCIPDijkstraImageToGraphFunctor_txx
#define __itkCIPDijkstraImageToGraphFunctor_txx

#include "itkCIPDijkstraImageToGraphFunctor.h"
#include "vnl/vnl_math.h"

namespace itk
{

template<typename TInputImage, typename TOutputGraph>
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::CIPDijkstraImageToGraphFunctor()
{
  this->m_UpperThreshold = itk::NumericTraits< PixelType >::max();
  this->m_LowerThreshold = itk::NumericTraits< PixelType >::NonpositiveMin();
  this->m_ExponentialCoefficient  = 50;
  this->m_ExponentialTimeConstant = 700;
  this->m_LinearBasedCostAssignment      = false;
  this->m_SigmoidBasedCostAssignment     = false;
  this->m_ExponentialBasedCostAssignment = true;
  this->m_SigmoidScale     = 5.0;
  this->m_SigmoidShift     = -800;
  this->m_SigmoidSteepness = 0.05;
}

template<typename TInputImage, typename TOutputGraph>
typename CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>::EdgeWeightType
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::GetEdgeWeight( IndexType idx1, IndexType idx2 )
{
  return static_cast< EdgeWeightType >( 0 );
}


template<typename TInputImage, typename TOutputGraph>
typename CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>::NodeWeightType
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::GetNodeWeight( IndexType idx1 )
{
  double pixelValue = static_cast< double >( this->GetInput()->GetPixel( idx1 ) );

  double nodeWeight;

  if ( this->m_LinearBasedCostAssignment )
    {
    nodeWeight = pixelValue;
    }
  else if ( this->m_ExponentialBasedCostAssignment )
    {
    nodeWeight = this->m_ExponentialCoefficient*std::exp( pixelValue/this->m_ExponentialTimeConstant );
    }
  else
    {
    nodeWeight = this->m_SigmoidScale/( 1.0 + std::exp( -this->m_SigmoidSteepness*( pixelValue - this->m_SigmoidShift ) ) );
    }

  return static_cast< NodeWeightType >( nodeWeight );
}


template<typename TInputImage, typename TOutputGraph>
bool
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::IsPixelANode( IndexType idx1 )
{
  if ( this->GetInput()->GetPixel( idx1 ) >= this->m_LowerThreshold &&
       this->GetInput()->GetPixel( idx1 ) <= this->m_UpperThreshold )
    {
    return true;
    }

  return false;
}


template<typename TInputImage, typename TOutputGraph>
bool 
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::IsAnEdge( IndexType idx1, IndexType idx2 )
{
  return true;
}


template<typename TInputImage, typename TOutputGraph>
void
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::NormalizeGraph( NodeImageType *image, OutputGraphType *graph )
{

}


template<typename TInputImage, typename TOutputGraph>
void
CIPDijkstraImageToGraphFunctor<TInputImage, TOutputGraph>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << " " << std::endl;
}


} // end namespace itk

#endif
