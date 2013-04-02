#ifndef __itkCIPHMMAirwayGraphTraits_h
#define __itkCIPHMMAirwayGraphTraits_h

#include "itkDefaultGraphTraits.h"

namespace itk
{

/** \class itkCIPHMMAirwayGraphTraits
 *  \brief Defines the graph traits needed to assign generation labels
 *  to airway particles using Hidden Markov Model (HMM).
 *
 *  This class extends the default node and edge structures in order
 *  to apply HMM analysis for labeling airway particles according to
 *  generation number. This class also includes node and edge fields
 *  needed by our implementation of Dijkstra's min-cost path algorithm.
 *  For nodes, this includes the 'Visited', 'Added', and
 *  'AccumulatedWeight' fields. For edges this inludes the
 *  'OptimalEdge' field. The node fields needed for generation
 *  labeling inlude 'Noise' and 'Generation'. Note that 'Noise' is a
 *  boolean value, and if it evaluates to 'true', the particle in the
 *  graph is assumed to be a noise particle, making the value stored
 *  in 'Generation' irrelevant.
 */


template < typename TWeight = double >
class CIPHMMAirwayGraphTraits : public DefaultGraphTraits< TWeight, TWeight >
{
public:
  typedef CIPHMMAirwayGraphTraits Self;
  typedef DefaultGraphTraits< TWeight, TWeight > Superclass;

  typedef typename Superclass::NodeIdentifierType   NodeIdentifierType;
  typedef typename Superclass::EdgeIdentifierType   EdgeIdentifierType;
  typedef typename Superclass::NodeWeightType       NodeWeightType;
  typedef typename Superclass::EdgeWeightType       EdgeWeightType;
  typedef std::vector<EdgeIdentifierType>           EdgeIdentifierContainerType;

  struct  NodeType;
  typedef NodeType* NodePointerType;

  struct NodeType
    {
      NodeIdentifierType                 Identifier;
      EdgeIdentifierContainerType        IncomingEdges;
      EdgeIdentifierContainerType        OutgoingEdges;
      NodeWeightType                     Weight;
      unsigned int                       ParticleID;
      std::map< unsigned char, double >  EmissionProbability;
    };

  struct  EdgeType;
  typedef EdgeType* EdgePointerType;

  struct EdgeType
    {
      EdgeIdentifierType                    Identifier;
      NodeIdentifierType                    SourceIdentifier;
      NodeIdentifierType                    TargetIdentifier;
      EdgeIdentifierType                    ReverseEdgeIdentifier;
      EdgeWeightType                        Weight;
      bool                                  OptimalEdge;
      std::vector< std::vector< double > >  TransitionMatrix;
    };
};


} // end namespace itk

#endif
