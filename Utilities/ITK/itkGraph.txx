/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkGraph.txx,v $
  Language:  C++
  Date:
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGraph_txx
#define __itkGraph_txx

#include "itkGraph.h"

namespace itk
{

template<typename TGraphTraits>
Graph<TGraphTraits>
::Graph()
{
  this->m_Nodes = NodeContainerType::New();
  this->m_Edges = EdgeContainerType::New();

  this->m_Edges->Initialize();
  this->m_Nodes->Initialize();
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::NodePointerType
Graph<TGraphTraits>
::CreateNewNode()
{
  NodePointerType node
    = &( this->m_Nodes->CreateElementAt( this->m_Nodes->Size() ) );
  node->Identifier = this->m_Nodes->Size()-1;
  node->Weight = static_cast<NodeWeightType>(1);
  return node;
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::NodePointerType
Graph<TGraphTraits>
::CreateNewNode( NodeWeightType Weight )
{
  NodePointerType node = this->CreateNewNode();
  node->Weight = Weight;
  return node;
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::CreateNewEdge()
{
  EdgePointerType edge;
    edge = &( this->m_Edges->CreateElementAt( this->m_Edges->Size() ) );
    edge->Identifier = this->m_Edges->Size()-1;
    edge->Weight     = static_cast<EdgeWeightType>( 1 );

  return edge;
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::CreateNewEdge( NodeIdentifierType SourceNodeId,
  NodeIdentifierType TargetNodeId )
{
  EdgePointerType edge = this->CreateNewEdge();
    edge->SourceIdentifier = SourceNodeId;
    edge->TargetIdentifier = TargetNodeId;
  
  this->GetNodePointer( SourceNodeId )->OutgoingEdges.push_back( edge->Identifier );
  this->GetNodePointer( TargetNodeId )->IncomingEdges.push_back( edge->Identifier );

  return edge;
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::CreateNewEdge( NodeIdentifierType SourceNodeId,
  NodeIdentifierType TargetNodeId, EdgeWeightType Weight )
{
  EdgePointerType edge = this->CreateNewEdge( SourceNodeId, TargetNodeId );
    edge->Weight = Weight;

  return edge;
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgeType&
Graph<TGraphTraits>
::GetReverseEdge( EdgeIdentifierType Id )
{
  return this->GetEdge( this->GetEdgePointer( Id )->ReverseEdgeIdentifier );
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgeType&
Graph<TGraphTraits>
::GetEdge( NodeIdentifierType SourceNodeId, NodeIdentifierType TargetNodeId )
{
  return *this->GetEdgePointer( SourceNodeId, TargetNodeId );
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::GetReverseEdgePointer( EdgeIdentifierType Id )
{
  return this->GetEdgePointer(
    this->GetEdgePointer( Id )->ReverseEdgeIdentifier );
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::GetReverseEdgePointer( EdgePointerType edge )
{
  return this->GetEdgePointer( edge->ReverseEdgeIdentifier );
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::GetEdgePointer( NodeIdentifierType SourceNodeId,
  NodeIdentifierType TargetNodeId )
{
  return this->GetEdgePointer( this->GetNodePointer(
    SourceNodeId ), this->GetNodePointer( TargetNodeId ) );
}

template<typename TGraphTraits>
typename Graph<TGraphTraits>::EdgePointerType
Graph<TGraphTraits>
::GetEdgePointer( NodePointerType SourceNode, NodePointerType TargetNode )
{
  typename EdgeIdentifierContainerType::iterator it;

  if ( SourceNode->OutgoingEdges.size()
    <= TargetNode->IncomingEdges.size() )
    {
    for ( it = SourceNode->OutgoingEdges.begin();
      it != SourceNode->OutgoingEdges.end(); it++ )
      {
      if ( TargetNode == this->GetTargetNodePointer( *it ) )
        {
        return this->GetEdgePointer( *it );
        }
      }
    }
  else
    {
    for ( it = TargetNode->IncomingEdges.begin();
      it != TargetNode->IncomingEdges.end(); it++ )
      {
      if ( SourceNode == this->GetSourceNodePointer( *it ) )
        {
        return this->GetEdgePointer( *it );
        }
      }
    }
  return NULL;
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::SetAllReverseEdges()
{
  EdgeIteratorType It;

  for ( It = this->m_Edges->Begin(); It != this->m_Edges->End(); ++It )
    {
    EdgePointerType edge = &It.Value();
    EdgePointerType reverse
      = this->GetEdgePointer( edge->TargetIdentifier, edge->SourceIdentifier );
    if ( reverse )
      {
      edge->ReverseEdgeIdentifier = reverse->Identifier;
      }
    }
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::Clear()
{
  this->m_Edges->Initialize();
  this->m_Nodes->Initialize();
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::SetEdgeContainer( EdgeContainerType *container )
{
  if ( this->m_Edges != container )
    {
    this->m_Edges = container;
    this->Modified();
    }
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::SetNodeContainer( NodeContainerType *container )
{
  if ( this->m_Nodes != container )
    {
    this->m_Nodes = container;
    this->Modified();
    }
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::Graft( const Self *data )
{
  if (  data  )
    {
    // Attempt to cast data to a graph
    const Self *graph;

    graph = dynamic_cast<const Self*>( data );

    if ( graph )
      {
      // Now copy anything remaining that is needed
      this->SetEdgeContainer(
        const_cast<typename Self::EdgeContainerType *>
        ( graph->GetEdgeContainer() )  );
      this->SetNodeContainer(
        const_cast<typename Self::NodeContainerType *>
        ( graph->GetNodeContainer() )  );
      }
    else
      {
      // pointer could not be cast back down
      itkExceptionMacro( "itk::Graph::Graft() cannot cast "
        << typeid( data ).name() << " to " << typeid( const Self * ).name() );
      }
  }
}

template<typename TGraphTraits>
Graph<TGraphTraits>
::~Graph()
{
  this->Clear();
}

template<typename TGraphTraits>
void
Graph<TGraphTraits>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Number of Nodes: " << this->m_Nodes->Size()  << std::endl;
  os << indent << "Number of Edges: " << this->m_Edges->Size()  << std::endl;
}

} // end namespace itk

#endif
