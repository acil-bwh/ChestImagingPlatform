/**
 *  \class itkCIPDijkstraGraphTraits.h
 *  \ingroup common
 *  \brief 
 *
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 *
 */
#ifndef __itkCIPDijkstraGraphTraits_h
#define __itkCIPDijkstraGraphTraits_h

#include "itkDefaultGraphTraits.h"
#include "itkIndex.h"

namespace itk
{

/**
 */

template < typename TWeight = short, unsigned int VImageDimension = 3 >
class CIPDijkstraGraphTraits : public DefaultGraphTraits< TWeight, TWeight >
{
public:
  typedef CIPDijkstraGraphTraits                  Self;
  typedef DefaultGraphTraits<TWeight, TWeight>    Superclass;

  typedef Index< VImageDimension >                           IndexType;
  typedef TWeight                                            NodeWeightType;
  typedef TWeight                                            EdgeWeightType;
  typedef typename Superclass::NodeIdentifierType            NodeIdentifierType;
  typedef typename Superclass::EdgeIdentifierType            EdgeIdentifierType;
  typedef typename Superclass::EdgeIdentifierContainerType   EdgeIdentifierContainerType;

  struct  NodeType;
  typedef NodeType* NodePointerType;

  struct NodeType
    {
      NodeIdentifierType          Identifier;
      EdgeIdentifierContainerType IncomingEdges;
      EdgeIdentifierContainerType OutgoingEdges;
      NodeWeightType              Weight;
      IndexType                   ImageIndex;
      bool                        Visited;
      bool                        Added;
      NodeWeightType              AccumulatedWeight;
    };

  struct  EdgeType;
  typedef EdgeType* EdgePointerType;

  struct EdgeType
  {
    EdgeIdentifierType Identifier;
    NodeIdentifierType SourceIdentifier;
    NodeIdentifierType TargetIdentifier;
    EdgeIdentifierType ReverseEdgeIdentifier;
    EdgeWeightType     Weight;
    bool               OptimalEdge;
  };

};

} // end namespace itk

#endif
