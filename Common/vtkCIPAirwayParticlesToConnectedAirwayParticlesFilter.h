/**
 *  \class vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter
 *  \ingroup common
 *  \brief This filter accepts poly data corresponding to
 *  airway particles and produces filtered poly data with reduced
 *  noise particles and with imposed topology.
 *
 *  The input poly data is assumed to be completely unfiltered. The
 *  poly data structure is also assumed to conform to the CIP
 *  conventions for airway particles described in cipConventions.h
 *
 *  The filter operates by first attempting to link particles together
 *  using both spatial proximity and orientation (using thresholds set
 *  with the 'SetParticleDistanceThreshold' and
 *  'SetParticleAngleThreshold' methods described below). Two particles are
 *  considered connected if they are within a specified distance from
 *  one another and if the angles formed between their minor
 *  eigenvectors and the vector connecting the particles' spatial
 *  locations are sufficiently small. Initially, if particles are
 *  found to be connected, bidirection edge links are created between
 *  the corresponding nodes in the graph representation.
 *
 *  After all particle-particle connections have been established,
 *  sub-graphs having low cardinality (as specified using the
 *  'SetSegmentCardinalityThreshold') are eliminated from further
 *  consideration, and those particles will not appear in the
 *  output. The remaining subgraphs are then considered for
 *  connectivity. Each leaf node of every subgraph is tested for
 *  connectivity to every other particle not in the leaf node's
 *  subgraph. A connection between the leaf node and another particle
 *  is established provided that the spatial distance between the two
 *  is sufficiently small (below a threshold specified with the 
 *  'SetSegmentDistanceThreshold' method). Additionally, the angle
 *  formed between the vector connecting the two particles and the
 *  leaf node's minor eigenvector must be below a certain threshold
 *  (specified using 'SetSegmentAngleThreshold'). Acknowledging that a
 *  given leaf node may potentially be connected to multiple particles
 *  in other subgraphs, a connection is only formed between the leaf
 *  node and the closest of the potential connecting particles. 
 *
 *  After all potenial connections between subgraphs have been tested,
 *  only the resulting subgraphs with sufficiently large cardinality
 *  (specified with the 'SetTreeCardinalityThreshold') are retained
 *  for further consideration. In an ideal scenario, all airway
 *  branches will be detected, and all branches will be appropriately
 *  connected in order to form a sinlge tree graph
 *  representation. This will in general not be the case,
 *  however. Noise branches may occur, true airway branches may not be
 *  detected, and the graph may actually exist as several disconnected
 *  subgraphs. In the following discussion, we will assume that we
 *  have a collection of disconnected subgraphs, with the ideal
 *  scenario of a fully connected tree being a special case.
 *
 *  The final step of the filter involves establishing a topology on
 *  the subgraphs that indicates direction of flow through the tree
 *  subgraphs. In order to do this, the most appropriate parent node
 *  in a given subtree must be determined. This is accomplished by
 *  considering each leaf node of a subgraph in turn and testing its
 *  suitability as the parent node. To test a given leaf node, all
 *  bidirectional links between nodes in the subgraph are substituted
 *  for unidirectional links emanating from the leaf node under
 *  consideration. The subgraph is then traversed starting at the leaf
 *  node, and at each branch point agreement with Murray's Law is
 *  evaluated. Murray's Law describes the relationship between the
 *  parent branch radius and the radii of the child branches. The
 *  relationship is given by r0^3 = r1^3 + r2^3, where r0 is the
 *  radius of the parent branch, and r1 and r2 are the radii of the
 *  child branches. Assuming some noise in our measurements of the
 *  radii and some slight departures from the ideal model in a given
 *  case, we can model agreement with this law with a zero-mean
 *  univariate Gaussian distribution: N(r0^3-r1^3-r2^3 | 0, variance),
 *  where the variance is learned from training data. We estimate the
 *  radii at a particle location using the following relationship
 *  between particle scale and airway radius (TODO). We can thus
 *  express the likelihood that a given leaf node is the parent node
 *  by the product of these Gaussians over all bifurcation points in
 *  the subgraph. The leaf node with the highest likelihood is chosen
 *  to be the parent node. (TODO: need to deal with the case of
 *  outliers: a single, noise-driven departure from Murray's Law can
 *  drive the entire likelihood term to near zero).
 *
 *  In the above discussion we have assumed that branch points are
 *  bifurcation points, as determined by a node being connected to
 *  three other nodes. However, in general a branch point may
 *  be connected more than three nodes due to noise branches, etc. To
 *  account for non-bifurcation branch scenarios, Murray's Law is
 *  tested for all pairs of child node nodes, and the pair yielding
 *  the highest agreement with Murrays Law is used in the likelihood
 *  calculation. 
 *
 *  After the parent node is established for each subgraph, the
 *  topology for the output poly data is created.

 *  $Date: 2012-09-17 18:32:23 -0400 (Mon, 17 Sep 2012) $
 *  $Revision: 268 $
 *  $Author: jross $
 *
 */

#ifndef __vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter_h
#define __vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter_h

#include "vtkPoints.h" //DEB 
#include "vtkPolyDataAlgorithm.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkMutableDirectedGraph.h" 
#include <map>
#include <vector>
#include "vtkCIPCommonConfigure.h"

 
class VTK_CIP_COMMON_EXPORT vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter : public vtkPolyDataAlgorithm 
{
public:
  vtkTypeMacro(vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
 
  static vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter *New();

  /** 
   *  The particle distance threshold (in mm) indicates how spatially
   *  close the two particles must be in order for them to be
   *  potentially linked in the graph representation (the particles
   *  must also be similarly oriented using a threshold specified
   *  using 'SetParticleAngleThreshold'). If the particles are not at
   *  least as close as the value specified, they will not be
   *  linked. The threshold to use should ideally be determined from
   *  training data. Both the particle distance threshold and the
   *  particle angle threshold should be just large enough so that all
   *  airway generation segments are connected in the training
   *  set. Note that this threshold should not be set just large
   *  enough to link neighboring particles. If it is set too large,
   *  more distant particles may potentially become connected, even
   *  though they have a common, intermediate, particle linking them
   *  in a chain. Such a secenario is not desirable as it could
   *  introduce cycles in the graph representation and break the
   *  assumption that the particles form a tree graph. Also note that
   *  the particles in the training data from which this threshold is
   *  determined should be generated with the same particles
   *  parameters intended for the test data.
   */
  vtkGetMacro( ParticleDistanceThreshold, double );
  vtkSetMacro( ParticleDistanceThreshold, double );

  /** 
   *  The particle angle threshold indicates how parallel the minor
   *  eigenvectors of two particles must be in order for them to be
   *  potentially linked in the graph representation (the particles
   *  must also be spatially close by specifying a value using the
   *  'ParticleDistanceThreshold'). The angle here is with respect to
   *  each of the particle's minor eigenvectors and the vector
   *  connecting the two particles. If either of these two vectors is
   *  larger than that specified with this method, the two particles
   *  will not be connected. The threshold to use should ideally be
   *  determined from training data. Both the particle distance
   *  threshold and the particle angle threshold should be just large
   *  enough so that all airway generation segments are connected in
   *  the training set.
   */
  /* vtkGetMacro( ParticleAngleThreshold, double ); */
  /* vtkSetMacro( ParticleAngleThreshold, double ); */

  vtkGetMacro( EdgeWeightAngleSigma, double ); 
  vtkSetMacro( EdgeWeightAngleSigma, double );
  

  /** 
   *  The segment distance threshold indicates how close (in mm) two
   *  connected segment subgraphs must be to one-another to be
   *  potentially considered as part of the same tree (sub) graph.
   */
  /* vtkGetMacro( SegmentDistanceThreshold, double ); */
  /* vtkSetMacro( SegmentDistanceThreshold, double ); */

  /** 
   *  After links have been established between particles (from the
   *  first round of connectivity tests), the algorithm attempts to
   *  link subgraphs. A (potential) link is established between a leaf
   *  node and a particle in some other subgraph provided that the
   *  distance between the two particles is sufficiently small
   *  (indicated with the 'SetSegmentDistanceThreshold') and provided
   *  that the angle formed between the vector connecting the two
   *  particles and the leaf node's minor eigenvector is sufficiently
   *  small (threshold value set with this method).
   *  
   */
  /* vtkGetMacro( SegmentAngleThreshold, double ); */
  /* vtkSetMacro( SegmentAngleThreshold, double ); */

  /** 
   *  The segment cardinality threshold determines the number of
   *  particles in a given connected segment that must be
   *  present in order for the segment to be preserved. Only segments
   *  having a number of particles equal to or greater than this
   *  threshold will be retained for the subsequent round of
   *  connectivity tests (in which segments are tested for
   *  connectivity to each other). This value should be determined
   *  from training data on which particles are genererated with the
   *  same parameters used for test data. For example, the
   *  inter-particle distance used for the training data should be the
   *  same as that used for the test data to ensure that the segment
   *  cardinality values determined on the training data apply to the
   *  test data. 
   */
  /* vtkGetMacro( SegmentCardinalityThreshold, unsigned int ); */
  /* vtkSetMacro( SegmentCardinalityThreshold, unsigned int ); */

  /** 
   *  The tree cardinality threshold determines the number of
   *  particles in a given connected tree subgraph that must be
   *  present in order for the subgraph to be preserved in the
   *  output. This value should be determined from training data from
   *  which particles are genererated with the same parameters used
   *  for test data. For example, the inter-particle distance used for
   *  the training data should be the same as that used for the test
   *  data to ensure that the tree cardinality values determined on
   *  the training data apply to the test data.
   */
  /* vtkGetMacro( TreeCardinalityThreshold, unsigned int ); */
  /* vtkSetMacro( TreeCardinalityThreshold, unsigned int ); */

protected:
  vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter();
  ~vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter();
 
  int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
 
private:
  vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter(const vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter&);  // Not implemented.
  void operator=(const vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter&);  // Not implemented.
 
  //  void InitializeDataStructureImageAndInternalInputPolyData( vtkPolyData* );

  double GetVectorMagnitude( double[3] );
  double GetAngleBetweenVectors( double[3], double[3], bool );

  /* bool EvaluateParticleConnectedness( unsigned int, unsigned int ); */
  /* void QueryNeighborhood( vtkIdType, unsigned int, vtkSmartPointer< vtkMutableDirectedGraph >, */
  /*                         std::map< vtkIdType, unsigned int >*, unsigned int, unsigned int, unsigned int ); */
  /* bool EvaluateGraphConnectedness( unsigned int, unsigned int ); */
  /* void AddSubGraphToSubTree( unsigned int, vtkSmartPointer< vtkMutableDirectedGraph >, std::map< vtkIdType, unsigned int >* ); */
  /* void AddSubGraphToSubTree( unsigned int, vtkSmartPointer< vtkMutableDirectedGraph >, std::map< vtkIdType, unsigned int >*, */
  /*                            unsigned int, unsigned int ); */
  /* bool AttemptAddSubGraphToSubTree( unsigned int, vtkSmartPointer< vtkMutableDirectedGraph >, std::map< vtkIdType, unsigned int >* ); */

  /* bool GetLeafNodeDirection( unsigned int, double* ); */

  /* double GetMinLeafNodeDistanceBetweenGraphs( unsigned int, unsigned int ); */

  /* void FillCompositeGraphWithSubTrees(); */

  void ViewGraphs( std::vector< vtkSmartPointer< vtkMutableDirectedGraph > >,
                   std::vector< std::map< vtkIdType, unsigned int > >, unsigned int );

  //  void DataExplorer();

  /* vtkSmartPointer< vtkImageData > DataStructureImage; */
  /* vtkSmartPointer< vtkPolyData >  InternalInputPolyData; */
  /* vtkSmartPointer< vtkPoints >    ConnectorPoints; //DEB */

  double EdgeWeightAngleSigma;

  //
  // The SubGraphsVec will contain all connected subGraphs (as
  // determined by connectivity tests using the
  // 'ParticleAngleThreshold' and the 'ParticleDistanceThreshold')
  // provided that the subGraphs have cardinality greater than or
  // equal to 'SegmentCardinalityThreshold'
  //
  //  std::vector< vtkSmartPointer< vtkMutableDirectedGraph > > SubGraphsVec;

  //
  // The SubTreesVec will contain all connected subTrees (as
  // determined by connectivity tests using the
  // 'SegmentAngleThreshold' and the 'SegmentDistanceThreshold')
  //
  //  std::vector< vtkSmartPointer< vtkMutableDirectedGraph > > SubTreesVec;

  //
  // The 'CompositeGraph' will contain the final graph corresponding
  // to all subTrees provided that each subTree has cardinality at
  // least as large as 'TreeCardinalityThreshold'
  //
  //  vtkSmartPointer< vtkMutableDirectedGraph > CompositeGraph;

  //
  // Each subgraph will consist of multiple nodes, each of which will
  // map to a specific particle. 'SubGraphNodeIDToParticleIDMapVec'
  // keeps track of the mapping, for each subGraph, between the subGraph's
  // node IDs to the corresponding particle IDs. A given vector index
  // for 'SubGraphNodeIDToParticleIDMapVec' and 'SubGraphsVec' points
  // to the same subGraph.
  //
  /* std::vector< std::map< vtkIdType, unsigned int > > SubGraphNodeIDToParticleIDMapVec; */

  /* std::vector< std::map< vtkIdType, unsigned int > > SubTreeNodeIDToParticleIDMapVec; */

  /* std::map< vtkIdType, unsigned int > CompositeGraphNodeIDToParticleIDMap; */

  //
  // After the particles have been tested for connectivity to each
  // other and membership to subGraphs has been established, we'll
  // want to collect the leaf nodes for each of the subGraphs. This
  // will be useful for the subsequent round of connectivity tests, in
  // which we attempt to link subGraphs together into subTrees. A link
  // can only be established between a subGraph leaf node and some
  // other node, so keep track of the leaf nodes will be helpful. Note
  // that this container only corresponds to leaf nodes of subGraphs
  // that have cardinality >= SegmentCardinalityThreshold.
  //
  //  std::vector< unsigned int > SubGraphLeafParticleIDs;

  //
  // After the subGraphs have been determined and the leaf nodes have
  // been designated, we'll test every leaf node for connection to
  // nodes in every other graph, establishing a connection provided
  // that the distance between the leaf node and the test node is at
  // least as close as 'SegmentDistanceThreshold' and provided that
  // the angle formed between the vector connecting the leaf node and
  // the test node and the leaf node's minor vector is no larger than
  // 'SegmentAngleThreshold'. If multiple connections are found, then
  // the connection corresponding to the smallest connecting distance
  // is used. 'ParticleIDToConnectingParticleIDMap' will keep track of
  // all pairs of particle IDs that connect to one another through
  // this process.
  //
  /* std::map< unsigned int, unsigned int > ParticleIDToConnectingParticleIDMap; */

  /* std::map< unsigned int, unsigned int > ParticleIDToSubGraphMap; */

  bool GetEdgeWeight( unsigned int, unsigned int, vtkSmartPointer< vtkPolyData >, double* );

  double       ParticleDistanceThreshold;
  /* double       ParticleAngleThreshold; */
  /* double       SegmentDistanceThreshold; */
  /* double       SegmentAngleThreshold; */
  /* double       InterParticleDistance; */
  /* double       DataImageSpacing; */
  /* unsigned int SegmentCardinalityThreshold; */
  /* unsigned int TreeCardinalityThreshold;  */
  unsigned int NumberOfPointDataArrays;
  unsigned int NumberInputParticles;
  /* unsigned int NumberInternalInputParticles; */
  /* unsigned int DataImageSize[3]; */
};
 
#endif
