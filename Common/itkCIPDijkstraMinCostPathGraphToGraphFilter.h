/**
 *  \class itkCIPDijkstraMinCostPathGraphToGraphFilter
 *  \ingroup common
 *  \brief Computes the minimum cost path between two specified points
 *  using Dijkstra's algorithm.
 *
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 *
 */

#ifndef __itkCIPDijkstraMinCostPathGraphToGraphFilter_h
#define __itkCIPDijkstraMinCostPathGraphToGraphFilter_h

#include "itkGraphToGraphFilter.h"

namespace itk
{

template <class TInputGraph, class TOutputGraph>
class ITK_EXPORT CIPDijkstraMinCostPathGraphToGraphFilter : public GraphToGraphFilter< TInputGraph, TOutputGraph >
{
public:
  /** Standard class typedefs. */
  typedef CIPDijkstraMinCostPathGraphToGraphFilter            Self;
  typedef GraphToGraphFilter< TInputGraph, TOutputGraph >  Superclass;
  typedef SmartPointer< Self >                             Pointer;
  typedef SmartPointer< const Self >                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( CIPDijkstraMinCostPathGraphToGraphFilter, GraphToGraphFilter );

  /** Some convenient typedefs. */
  typedef TInputGraph                                   InputGraphType;
  typedef TOutputGraph                                  OutputGraphType;
  typedef typename InputGraphType::Pointer              InputGraphPointer;
  typedef typename InputGraphType::NodeIdentifierType   InputNodeIdentifierType;
  typedef typename OutputGraphType::NodeIdentifierType  OutputNodeIdentifierType;
  typedef typename InputGraphType::NodeIterator         InputNodeIteratorType;
  typedef typename InputGraphType::EdgeIterator         InputEdgeIteratorType;
  typedef typename InputGraphType::NodeWeightType       InputNodeWeightType;
  typedef typename InputGraphType::NodeType             InputNodeType;
  typedef typename InputGraphType::NodePointerType      InputNodePointerType;
  typedef typename OutputGraphType::NodePointerType     OutputNodePointerType;

  itkSetMacro( StartNode, InputNodeIdentifierType );
  itkGetMacro( StartNode, InputNodeIdentifierType );

  itkSetMacro( EndNode, InputNodeIdentifierType );
  itkGetMacro( EndNode, InputNodeIdentifierType );
  

protected:
  CIPDijkstraMinCostPathGraphToGraphFilter();
  ~CIPDijkstraMinCostPathGraphToGraphFilter() {};

  void GenerateData() override;

private:
  CIPDijkstraMinCostPathGraphToGraphFilter( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  bool GetIDOfLowestCostUnvisited( InputNodeIdentifierType&, std::vector< InputNodeIdentifierType >& );

  InputNodeIdentifierType m_StartNode;
  InputNodeIdentifierType m_EndNode;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPDijkstraMinCostPathGraphToGraphFilter.txx"
#endif

#endif
