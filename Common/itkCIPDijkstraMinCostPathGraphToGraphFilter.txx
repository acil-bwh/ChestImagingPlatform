/**
 *
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 *
 */

#ifndef __itkCIPDijkstraMinCostPathGraphToGraphFilter_txx
#define __itkCIPDijkstraMinCostPathGraphToGraphFilter_txx

#include "itkCIPDijkstraMinCostPathGraphToGraphFilter.h"


namespace itk
{

/**
 *
 */
template < class TInputGraph, class TOutputGraph >
CIPDijkstraMinCostPathGraphToGraphFilter< TInputGraph, TOutputGraph >
::CIPDijkstraMinCostPathGraphToGraphFilter()
{
}


template < class TInputGraph, class TOutputGraph >
void
CIPDijkstraMinCostPathGraphToGraphFilter< TInputGraph, TOutputGraph >
::GenerateData()
{
  InputNodeIteratorType nIt( this->GetInput() );

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    nIt.Get().AccumulatedWeight = itk::NumericTraits< InputNodeWeightType >::max();
    nIt.Get().Visited = false;
    nIt.Get().Added   = false;

    ++nIt;
    }

  InputEdgeIteratorType eIt( this->GetInput() );

  eIt.GoToBegin();
  while ( !eIt.IsAtEnd() )
    {
    eIt.Get().OptimalEdge = false;
    
    ++eIt;
    }

  std::vector< InputNodeIdentifierType > nodeIDVec;

  bool endNodeVisited = false;
  
  this->GetInput()->GetNode( this->m_StartNode ).Added = true;
  this->GetInput()->GetNode( this->m_StartNode ).AccumulatedWeight = this->GetInput()->GetNode( this->m_StartNode ).Weight;

  typename InputGraphType::NodeIdentifierType visitingNodeID = this->m_StartNode;

  do 
    {
    if ( visitingNodeID == this->m_EndNode )
      {
      endNodeVisited = true;
      }

    //-------
    // Get all the outgoing edges from the current node
    //
    typename InputGraphType::NodePointerType              visitingNodePointer = this->GetInput()->GetNodePointer( visitingNodeID );
    typename InputGraphType::EdgeIdentifierContainerType  outgoingEdgeVec     = this->GetInput()->GetOutgoingEdges( visitingNodePointer );
    
    //-------
    // Set the visiting node to 'Visited'.
    //
    this->GetInput()->GetNode( visitingNodeID ).Visited = true;
    
    //-------
    // Now investigate all the nodes that the visiting node points
    // to. If the visiting node accumulated value plus the pointed to
    // node value is less than the pointed to's accumulated value,
    // replace that accumulated value with this sum.  Otherwise, set
    // the pointed to's 'OptimalEdge' value to false
    //
    InputNodeWeightType visitingAccumulatedWeight = visitingNodePointer->AccumulatedWeight;

    for ( int i=0; i<outgoingEdgeVec.size(); i++ )
      {
      InputNodeType targetNode = this->GetInput()->GetTargetNode( outgoingEdgeVec[i] );
              
      InputNodeIdentifierType targetNodeID               = this->GetInput()->GetNodeIdentifier( targetNode );
        
      InputNodeWeightType     pointedToWeight            = targetNode.Weight;
      InputNodeWeightType     pointedToAccumulatedWeight = targetNode.AccumulatedWeight;
      InputNodeWeightType     weightSum                  = static_cast< InputNodeWeightType >( visitingAccumulatedWeight + pointedToWeight );

      if ( !targetNode.Added )
        {
        this->GetInput()->GetNode( targetNodeID ).Added = true;

        nodeIDVec.push_back( targetNodeID );
        }

      if ( weightSum < pointedToAccumulatedWeight )
        {
        //-------
        // We have found the best current path to the target node.
        // There might be an obsolete optimal path that we'll have
        // to break.  Look through all the edges that sink into the
        // target node and make sure those edges are set to
        // non-optimal.  
        //
        typename InputGraphType::NodePointerType targetNodePointer = this->GetInput()->GetNodePointer( targetNodeID );

        typename InputGraphType::EdgeIdentifierContainerType tempIncomingEdgeVec = this->GetInput()->GetIncomingEdges( targetNodePointer );
        
        for ( int e=0; e<tempIncomingEdgeVec.size(); e++ )
          {
          this->GetInput()->GetEdge( tempIncomingEdgeVec[e] ).OptimalEdge = false;
          }
        
        //-------
        // Now set the optimal edge and weight sum
        //
        this->GetInput()->GetNode( targetNodeID ).AccumulatedWeight = weightSum;
        this->GetInput()->GetEdge( visitingNodeID, targetNodeID ).OptimalEdge = true;
        }
      }    
    }
  while ( this->GetIDOfLowestCostUnvisited( visitingNodeID, nodeIDVec )  );

  //-------
  // Now that we have the optimal edges determined, we can back-track
  // from the end node to the start node to find all the indices along
  // the minimum cost path between the endpoints
  //  
  typename InputGraphType::NodeIdentifierType currentNodeID = this->m_EndNode;

  //-------
  // Create a new node with the proper weight.  Set its image index,
  // and make it point from the current node to the previous node.
  //
  OutputNodeIdentifierType currentOutputNodeID;
  OutputNodeIdentifierType previousOutputNodeID;

  OutputNodePointerType outputEndNodePtr = this->GetOutput()->CreateNewNode();

  currentOutputNodeID = this->GetOutput()->GetNodeIdentifier( outputEndNodePtr );

  this->GetOutput()->GetNode( currentOutputNodeID ).ImageIndex = this->GetInput()->GetNode( currentNodeID ).ImageIndex;
  this->GetOutput()->GetNode( currentOutputNodeID ).Weight     = this->GetInput()->GetNode( currentNodeID ).Weight;

  previousOutputNodeID = currentOutputNodeID;

  bool arrivedAtStart = false;
  while ( !arrivedAtStart )
    {
    //-------
    // Get all the incoming edges to the current node
    //
    typename InputGraphType::NodePointerType currentNodePointer = this->GetInput()->GetNodePointer( currentNodeID );
    typename InputGraphType::EdgeIdentifierContainerType incomingEdgeVec = this->GetInput()->GetIncomingEdges( currentNodePointer );

    //-------
    // Find out which one is the optimal edge
    //
    bool optimalEdgeFound = false;
    for ( int i=0; i<incomingEdgeVec.size(); i++ )
      {
      if ( this->GetInput()->GetEdge( incomingEdgeVec[i] ).OptimalEdge == true )
        {
        optimalEdgeFound = true;

        //-------
        // Determine the optimal edge's source node
        //
        typename InputGraphType::NodeType sourceNode = this->GetInput()->GetSourceNode( incomingEdgeVec[i] );

        //-------
        // Now create the new output node and assign to it the proper
        // image index and weight.  Also establish a link between the
        // current and previous nodes
        //
        OutputNodePointerType outputNodePtr = this->GetOutput()->CreateNewNode();
        currentOutputNodeID = this->GetOutput()->GetNodeIdentifier( outputNodePtr );

        this->GetOutput()->GetNode( currentOutputNodeID ).ImageIndex = sourceNode.ImageIndex;
        this->GetOutput()->GetNode( currentOutputNodeID ).Weight = sourceNode.Weight;
        this->GetOutput()->CreateNewEdge( currentOutputNodeID, previousOutputNodeID );

        previousOutputNodeID = currentOutputNodeID;

        //-------
        // Set currentNodeID to be the ID of the source node
        //
        currentNodeID = this->GetInput()->GetNodeIdentifier( sourceNode );

        //-------
        // Is the current node the startNode?  If so, set arrivedAtStart
        // node to true
        //
        if ( this->GetInput()->GetNodeIdentifier( sourceNode ) == this->m_StartNode )
          {
          arrivedAtStart = true;
          }

        break;
        }
      }
    }  
}


template < class TInputGraph, class TOutputGraph >
bool
CIPDijkstraMinCostPathGraphToGraphFilter< TInputGraph, TOutputGraph >
::GetIDOfLowestCostUnvisited( InputNodeIdentifierType& nodeID, std::vector< InputNodeIdentifierType >&  nodeIDVec )
{
  bool nodeFound = false;

  InputNodeWeightType minWeight = itk::NumericTraits< InputNodeWeightType >::max();

  int whichElement;

  for ( int i=0; i<nodeIDVec.size(); i++ )
    {
    if ( !this->GetInput()->GetNode( nodeIDVec[i] ).Visited )
      {
      if ( this->GetInput()->GetNode( nodeIDVec[i] ).AccumulatedWeight < minWeight )
        {
        minWeight = this->GetInput()->GetNode( nodeIDVec[i] ).AccumulatedWeight;

        nodeID = nodeIDVec[i];
        
        nodeFound = true;
        
        whichElement = i;
        }
      }
    }

  if ( nodeFound )
    {
    typename std::vector< InputNodeIdentifierType >::iterator vIt = nodeIDVec.begin() + whichElement;

    nodeIDVec.erase( vIt );
    }

  return nodeFound;
}



} // end namespace itk

#endif
