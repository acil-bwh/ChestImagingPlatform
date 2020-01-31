/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkGraph.h,v $
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
#ifndef __itkGraph_h
#define __itkGraph_h

#include "itkDataObject.h"
#include "itkObjectFactory.h"
#include "itkVectorContainer.h"

namespace itk
{

/** \class Graph
 * \brief Basic graph class.
 *
 * \par
 * Directional graph class which has basic functionality.  It is
 * templated over TGraphTraits which contain the definitions for
 * nodes and edges (see itkDefaultGraphTraits.h for the basic
 * concept).  This class maintains two container types - one for
 * the nodes and one for the edges.  It also defines two basic
 * iterator classes which are simply interfaces to the iterator
 * types already defined for the container type.
 *
 * \ingroup GraphObjects
 * \ingroup DataRepresentation
 */

template <typename TGraphTraits>
class ITK_EXPORT Graph : public DataObject
{
public:
  /** Standard class typedefs. */
  typedef Graph                       Self;
  typedef DataObject                  Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Standard part of every itk Object. */
  itkTypeMacro( Graph, DataObject );

  /** Hold on to the type information specified by the template parameters. */
  typedef TGraphTraits                                   GraphTraitsType;
  typedef typename GraphTraitsType::NodeType             NodeType;
  typedef typename GraphTraitsType::EdgeType             EdgeType;
  typedef typename GraphTraitsType::NodePointerType      NodePointerType;
  typedef typename GraphTraitsType::EdgePointerType      EdgePointerType;
  typedef typename GraphTraitsType::NodeIdentifierType   NodeIdentifierType;
  typedef typename GraphTraitsType::EdgeIdentifierType   EdgeIdentifierType;
  typedef typename GraphTraitsType::NodeWeightType       NodeWeightType;
  typedef typename GraphTraitsType::EdgeWeightType       EdgeWeightType;

  typedef typename GraphTraitsType::EdgeIdentifierContainerType
                                              EdgeIdentifierContainerType;

  typedef VectorContainer<unsigned, NodeType>        NodeContainerType;
  typedef typename NodeContainerType::Iterator       NodeIteratorType;
  typedef typename NodeContainerType::ConstIterator  NodeConstIteratorType;
  typedef VectorContainer<unsigned, EdgeType>        EdgeContainerType;
  typedef typename EdgeContainerType::Iterator       EdgeIteratorType;
  typedef typename EdgeContainerType::ConstIterator  EdgeConstIteratorType;

  /** Return the total number of nodes. */
  unsigned int GetTotalNumberOfNodes()
    { return m_Nodes->Size(); }
  /** Return the total number of edges. */
  unsigned int GetTotalNumberOfEdges()
    { return m_Edges->Size(); }

  /** Clear the graph */
  void Clear();

  /** Create new nodes */
  NodePointerType CreateNewNode();
  NodePointerType CreateNewNode( NodeWeightType );

  /** Create new edges */
  EdgePointerType CreateNewEdge();
  EdgePointerType CreateNewEdge( NodeIdentifierType, NodeIdentifierType );
  EdgePointerType CreateNewEdge( NodeIdentifierType, NodeIdentifierType, EdgeWeightType);
  EdgePointerType CreateNewEdge( NodePointerType SourceNode, NodePointerType TargetNode )
    {
    return this->CreateNewEdge( SourceNode->Identifier, TargetNode->Identifier );
    }
  EdgePointerType CreateNewEdge( NodePointerType SourceNode, NodePointerType TargetNode, EdgeWeightType w )
    {
    return this->CreateNewEdge( SourceNode->Identifier, TargetNode->Identifier, w );
    }
  EdgePointerType CreateNewEdge( NodeType SourceNode, NodeType TargetNode)
    {
    return this->CreateNewEdge( SourceNode.Identifier, TargetNode.Identifier);
    }
  EdgePointerType CreateNewEdge( NodeType SourceNode, NodeType TargetNode, EdgeWeightType w )
    {
    return this->CreateNewEdge( SourceNode.Identifier, TargetNode.Identifier, w );
    }

  /** Graph utility functions */
  /** Get Nodes/Edges         */
  NodeType& GetNode( NodeIdentifierType Id )
    { return m_Nodes->ElementAt( Id ); }
  EdgeType& GetEdge( EdgeIdentifierType Id )
    { return m_Edges->ElementAt( Id ); }
  EdgeType& GetReverseEdge( EdgeIdentifierType );
  NodeType& GetSourceNode( EdgePointerType Edge )
    { return this->GetNode( Edge->SourceIdentifier ); }
  NodeType& GetSourceNode( EdgeType Edge )
    { return this->GetNode( Edge.SourceIdentifier ); }
  NodeType& GetSourceNode( EdgeIdentifierType Id )
    { return this->GetSourceNode( this->GetEdgePointer( Id ) ); }
  NodeType& GetTargetNode( EdgePointerType Edge )
    { return this->GetNode( Edge->TargetIdentifier ); }
  NodeType& GetTargetNode( EdgeType Edge )
    { return this->GetNode( Edge.TargetIdentifier ); }
  NodeType& GetTargetNode( EdgeIdentifierType Id )
    { return this->GetTargetNode( this->GetEdgePointer( Id ) ); }
  EdgeType& GetEdge( NodeIdentifierType, NodeIdentifierType );

  NodePointerType GetNodePointer( NodeIdentifierType Id )
    { return &m_Nodes->ElementAt( Id ); }
  EdgePointerType GetEdgePointer( EdgeIdentifierType Id )
    { return &m_Edges->ElementAt( Id ); }
  EdgePointerType GetReverseEdgePointer( EdgeIdentifierType );
  EdgePointerType GetReverseEdgePointer( EdgePointerType );
  NodePointerType GetSourceNodePointer( EdgePointerType Edge )
    { return this->GetNodePointer( Edge->SourceIdentifier ); }
  NodePointerType GetSourceNodePointer( EdgeIdentifierType Id )
    { return this->GetSourceNodePointer( this->GetEdgePointer( Id ) ); }
  NodePointerType GetTargetNodePointer( EdgePointerType Edge )
    { return this->GetNodePointer( Edge->TargetIdentifier ); }
  NodePointerType GetTargetNodePointer( EdgeIdentifierType Id )
    { return this->GetTargetNodePointer( this->GetEdgePointer( Id ) ); }
  EdgePointerType GetEdgePointer( NodeIdentifierType, NodeIdentifierType );
  EdgePointerType GetEdgePointer( NodePointerType, NodePointerType );

  /** Get Node/Edge Identifiers */
  NodeIdentifierType GetNodeIdentifier( NodePointerType node )
    { return node->Identifier; }
  EdgeIdentifierType GetEdgeIdentifier( EdgePointerType edge )
    { return edge->Identifier; }
  NodeIdentifierType GetNodeIdentifier( NodeType node )
    { return node.Identifier; }
  EdgeIdentifierType GetEdgeIdentifier( EdgeType edge )
    { return edge.Identifier; }

  /** Get/Set/Add Node/Edge weights */
  NodeWeightType GetNodeWeight( NodePointerType Node )
    { return Node->Weight; }
  EdgeWeightType GetEdgeWeight( EdgePointerType Edge )
    { return Edge->Weight; }
  NodeWeightType GetNodeWeight( NodeType Node )
    { return Node.Weight; }
  EdgeWeightType GetEdgeWeight( EdgeType Edge )
    { return Edge.Weight; }
  NodeWeightType GetNodeWeight( NodeIdentifierType Id )
    { return this->GetNodePointer( Id )->Weight; }
  EdgeWeightType GetEdgeWeight( EdgeIdentifierType Id )
    { return this->GetEdgePointer( Id )->Weight; }
  void SetNodeWeight( NodePointerType Node, NodeWeightType w )
    { Node->Weight = w; }
  void SetEdgeWeight( EdgePointerType Edge, EdgeWeightType w )
    { Edge->Weight = w; }
  void SetNodeWeight( NodeType Node, NodeWeightType w )
    { Node.Weight = w; }
  void SetEdgeWeight( EdgeType Edge, EdgeWeightType w )
    { Edge.Weight = w; }
  void SetNodeWeight( NodeIdentifierType Id, NodeWeightType w )
    { this->GetNodePointer( Id )->Weight = w; }
  void SetEdgeWeight( EdgeIdentifierType Id, EdgeWeightType w )
    { this->GetEdgePointer( Id )->Weight = w; }
  void AddNodeWeight( NodePointerType Node, NodeWeightType w )
    { Node->Weight += w; }
  void AddEdgeWeight( EdgePointerType Edge, EdgeWeightType w )
    { Edge->Weight += w; }
  void AddNodeWeight( NodeType Node, NodeWeightType w )
    { Node.Weight += w; }
  void AddEdgeWeight( EdgeType Edge, EdgeWeightType w )
    { Edge.Weight += w; }
  void AddNodeWeight( NodeIdentifierType Id, NodeWeightType w )
    { this->GetNodePointer( Id )->Weight += w; }
  void AddEdgeWeight( EdgeIdentifierType Id, EdgeWeightType w )
    { this->GetEdgePointer( Id )->Weight += w; }

  /** Get edges to adjacent nodes */
  EdgeIdentifierContainerType GetOutgoingEdges( NodePointerType Node )
    { return Node->OutgoingEdges; }
  EdgeIdentifierContainerType GetIncomingEdges( NodePointerType Node )
    { return Node->IncomingEdges; }

  /**
   * After creating all the edges, this function associates each
   * edge with it's reverse edge ( if it exists ).
   */
  virtual void SetAllReverseEdges();

  void ChangeNodeWeight( NodeIdentifierType Id, NodeWeightType W )
    { this->GetNodePointer( Id )->Weight = W; }
  void ChangeEdgeWeight( EdgeIdentifierType Id, EdgeWeightType W )
    { this->GetEdgePointer( Id )->Weight = W; }

  NodeContainerType* GetNodeContainer()
    { return this->m_Nodes.GetPointer(); }
  const NodeContainerType* GetNodeContainer() const
    { return this->m_Nodes.GetPointer(); }
  void SetNodeContainer( NodeContainerType * );
  EdgeContainerType* GetEdgeContainer()
    { return this->m_Edges.GetPointer(); }
  const EdgeContainerType* GetEdgeContainer() const
    { return this->m_Edges.GetPointer(); }
  void SetEdgeContainer( EdgeContainerType * );

  void Graft( const Self * );

  /**
   * Define the node/edge iterators which are simple
   * wrappers for existing iterators of the wrapper
   * class.
   */

  friend class NodeIterator;
  friend class EdgeIterator;

  class NodeIterator
    {
    public:
    NodeIterator( Graph* graph ) 
      { this->m_Graph = graph; }
    ~NodeIterator() {}

    /** Iterator-related functions */
    void GoToBegin(void)
      { this->m_NodeIterator = this->m_Graph->m_Nodes->Begin(); }
    bool IsAtEnd(void)
      { return ( this->m_NodeIterator == this->m_Graph->m_Nodes->End() ); }
    void operator++()
      { m_NodeIterator++; }
    NodePointerType GetPointer( void )
      { return &this->m_NodeIterator.Value(); }
    NodeType& Get( void )
      { return this->m_NodeIterator.Value(); }
    unsigned long GetIdentifier( void )
      { return this->m_NodeIterator.Index(); }
  
    private:
    Graph* m_Graph;
    NodeIteratorType m_NodeIterator;
    };

  class EdgeIterator
    {
    public:
    EdgeIterator( Graph* graph ) 
      { this->m_Graph = graph; }
    ~EdgeIterator() {}

    /** Iterator-related functions */
    void GoToBegin( void )
      { this->m_EdgeIterator = this->m_Graph->m_Edges->Begin(); }
    bool IsAtEnd( void )
      { return ( this->m_EdgeIterator == this->m_Graph->m_Edges->End() ); }
    void operator++() { m_EdgeIterator++; }
    EdgePointerType GetPointer(void)
      { return &this->m_EdgeIterator.Value(); }
    EdgeType& Get( void )
      { return this->m_EdgeIterator.Value(); }
    unsigned long GetIdentifier( void )
      { return this->m_EdgeIterator.Index(); }

    private:
    Graph* m_Graph;
    EdgeIteratorType m_EdgeIterator;
    };

protected:
  /** Constructor for use by New() method. */
  Graph();
  ~Graph();
  virtual void PrintSelf( std::ostream& os, Indent indent ) const override;

private:
  Graph( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  typename EdgeContainerType::Pointer m_Edges;
  typename NodeContainerType::Pointer m_Nodes;
}; // End Class: Graph

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGraph.txx"
#endif

#endif
