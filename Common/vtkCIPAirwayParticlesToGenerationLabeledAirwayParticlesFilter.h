/**
 *  \class vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter
 *  \ingroup common
 *  \brief This filter...
 *
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 */

#ifndef __vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter_h
#define __vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter_h

#include "vtkPoints.h" //DEB 
#include "vtkPolyDataAlgorithm.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkMutableDirectedGraph.h" 
#include "vtkMutableUndirectedGraph.h" 
#include <map>
#include <vector>
#include "vtkCIPCommonConfigure.h"

 
class VTK_CIP_COMMON_EXPORT vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter : public vtkPolyDataAlgorithm 
{
public:
  vtkTypeMacro(vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
 
  static vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter *New();

  /** 
   *  The particle distance threshold (in mm) indicates how spatially
   *  close the two particles must be in order for them to be
   *  potentially linked in the minimum spanning tree representation.
   */
  vtkGetMacro( ParticleDistanceThreshold, double );
  vtkSetMacro( ParticleDistanceThreshold, double );

  /** The weight of the edge between two particles used to construct the graph that
   *  is passed to the min spanning tree algorithm is a function of the distance 
   *  between two particles and the angle between their minor eigenvectors. This 
   *  parameter controls the effect of the angle contribution.
   */
  vtkGetMacro( EdgeWeightAngleSigma, double ); 
  vtkSetMacro( EdgeWeightAngleSigma, double );

  /** Optionally specify the spherical radius region of interest
   *  over which contributions to the kernel density estimation
   *  are made. Only atlas particles that are within this physical
   *  distance will contribute to the estimate. By default, all
   *  atlas particles will contribute to the estimate.
   */
  vtkGetMacro( KernelDensityEstimationROIRadius, double ); 
  vtkSetMacro( KernelDensityEstimationROIRadius, double );

  /** You must specify at least one airway-generation labeled atlas for the
   *  filter to work properly. An airway generation labeled atlas is a 
   *  particles data set that has field data array field named 'ChestType' that,
   *  for each particle, has a correctly labeled airway generation label.
   *  Labeling must conform to the standards set forth in 'cipConventions.h'.
   *  The atlas must be in the same coordinate frame as the input dataset that
   *  is to be labeled. Multiple atlases may be specified. These atlases are 
   *  used to compute the emission probabilities (see descriptions of the HMM
   *  algorithm) using kernel density estimation.
   */
  void AddAirwayGenerationLabeledAtlas( const vtkSmartPointer< vtkPolyData > );
  
  /** Set the particle of the tree root node if known. Setting this variable
   *  explicitly makes it unnecessary to search for which leaf node is the
   *  most probable root node. Assumes a single connected tree. The particle
   *  root node is most typically the trachea "leaf" node.
   */
  void SetParticleRootNodeID( unsigned int );

  /** The airway label assignment algorithm works in one of two modes: KDE or
   *  HMTM. If the mode is set to HMTM (Hidden Markov Tree Model), the complete
   *  probabilistic model will be applied. By default this is set to true. */
  void SetModeToHMTM();

  /** The airway label assignment algorithm works in one of two modes: KDE or
   *  HMTM. If the mode is set to KDE (kernel density estimation), airway label 
   *  assignments are made using KDE based classification. This is equivalent to
   *  only considering the emission probabilities used in the complete HMTM. By
   *  default this is set to false. */
  void SetModeToKDE();

protected:
  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter();
  ~vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter();
 
  int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
 
private:
  // In general, the particles may exist as a set of subgraphs.
  // We will need to run the HMM generation labeler independently
  // for each subgraph. This structure will facilitate keeping 
  // track of the various subgraphs.
  struct SUBGRAPH
  {
    vtkSmartPointer< vtkMutableUndirectedGraph>  undirectedGraph;
    std::vector< vtkIdType >                     leafNodeIDs;
    std::map< vtkIdType, unsigned int >          nodeIDToParticleIDMap;
    std::map< unsigned int, vtkIdType >          particleIDToNodeIDMap;
  };

  // The structure 'TRANSITIONPROBABILITY' is used to facilitate
  // computation of transition probabilities. By default,
  // all transition probabilities are set to 0.0 (i.e. they are
  // all impossible). 
  struct TRANSITIONPROBABILITY
  {
    unsigned char sourceState;
    unsigned char targetState;
    double        probability;
  };

  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter(const vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter&);  // Not implemented.
  void operator=(const vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter&);  // Not implemented.

  double GetTransitionProbability( unsigned int, unsigned int, unsigned char, unsigned int, vtkSmartPointer< vtkPolyData > );
  double ComputeGenerationLabelsFromTrellisGraph( vtkSmartPointer< vtkMutableDirectedGraph >, std::map< unsigned int, unsigned char >* );

  void InitializeEmissionProbabilites( vtkSmartPointer< vtkPolyData > );
  void InitializeSubGraphs( vtkSmartPointer< vtkMutableUndirectedGraph >, vtkSmartPointer< vtkPolyData > );
  void InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > );
  void InitializeAirwayGenerationAssignments( unsigned int );
  void GetTrellisGraphFromSubgraph( SUBGRAPH*, vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph >, vtkSmartPointer< vtkPolyData > );
  void AddStateNodesToTrellisGraph( vtkSmartPointer< vtkMutableDirectedGraph >, std::vector< vtkIdType >, vtkIdType,
				    SUBGRAPH*, std::map< unsigned int, bool >* );
  void UpdateAirwayGenerationAssignments( std::map< unsigned int, unsigned char >* );
  bool UpdateTrellisGraphWithViterbiStep( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph > );
  void BackTrack( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph >, std::map< unsigned int, unsigned char >*, double* );

  double EdgeWeightAngleSigma;

  bool IsNonRootLeafNode( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph > );
  bool GetEdgeWeight( unsigned int, unsigned int, vtkSmartPointer< vtkPolyData >, double* );

  vtkSmartPointer< vtkMutableUndirectedGraph > MinimumSpanningTree;

  std::map< unsigned int, unsigned char >                    ParticleIDToAirwayGenerationMap;
  std::map< unsigned int, std::map<unsigned char, double> >  ParticleIDToEmissionProbabilitiesMap;
  std::vector< SUBGRAPH >                                    Subgraphs;
  std::vector< unsigned char >                               States;
  std::vector< TRANSITIONPROBABILITY >                       TransitionProbabilities;
  std::vector< TRANSITIONPROBABILITY >                       TransitionProbabilityPriors;
  std::vector< vtkSmartPointer< vtkPolyData > >              AirwayGenerationLabeledAtlases;

  // For computation of the emissision probabilities, we use kernel density estimation.
  // Specifically, we use exponential distributions for both distance and angle, and
  // we use a Gaussian for scale. The parameters for these distributions are defined
  // below and set in the constructor. The values chosen are based on training data.
  double EmissionDistanceLambda;
  double EmissionScaleMu;
  double EmissionScaleSigma;
  double EmissionAngleLambda;

  // The likelihoods used for computing transition probabilities depend on whether the
  // "transition" in question is a "same state" "transition" or a different state
  // transition. For same state transitions, we model the scale difference term as a 
  // gaussian. For the angle term, we use an exponential distribution.
  double SameTransitionScaleMu;
  double SameTransitionScaleSigma;
  double SameTransitionAngleLambda;

  // The "different state" transition distributions are more complicated. For the scale
  // difference distribution, we use a two component GMM. For the angle difference 
  // distribution, we use a piece-wise linear function (an increasing linear function
  // for part -- up to 20 degrees -- of the domain and a decreasing linear component 
  // for the remainder of the domain -- from 20 to 90 degrees).
  double DiffTransitionScaleMu1;
  double DiffTransitionScaleSigma1;
  double DiffTransitionScaleWeight1;
  double DiffTransitionScaleMu2;
  double DiffTransitionScaleSigma2;
  double DiffTransitionScaleWeight2;
  double DiffTransitionAngleSlope1;
  double DiffTransitionAngleSlope2;
  double DiffTransitionAngleIntercept1;
  double DiffTransitionAngleIntercept2;

  bool         HMTMMode;
  int          ParticleRootNodeID;
  double       ParticleDistanceThreshold;
  double       KernelDensityEstimationROIRadius;
  double       NoiseProbability;
  unsigned int NumberOfPointDataArrays;
  unsigned int NumberInputParticles;
  unsigned int NumberOfStates;
};
 
#endif
