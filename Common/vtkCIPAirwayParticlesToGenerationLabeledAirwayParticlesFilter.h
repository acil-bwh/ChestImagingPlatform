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

 
class vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter : public vtkPolyDataAlgorithm 
{
public:
  vtkTypeMacro(vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
 
  static vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter *New();

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
  
  /** For each airway generation, specify the scale standard deviation
   *  to be used in the Gaussian-kernel density estimation. */
  void SetScaleStandardDeviation( unsigned char, double );

  /** For each airway generation, specify the distance standard deviation
   *  to be used in the Gaussian-kernel density estimation. */
  void SetDistanceStandardDeviation( unsigned char, double );

  /** For each airway generation, specify the angle standard deviation
   *  to be used in the Gaussian-kernel density estimation. */
  void SetAngleStandardDeviation( unsigned char, double );

  /** For each airway generation, specify the scale mean
   *  to be used in the Gaussian-kernel density estimation. */
  void SetScaleMean( unsigned char, double );

  /** For each airway generation, specify the distance mean
   *  to be used in the Gaussian-kernel density estimation. */
  void SetDistanceMean( unsigned char, double );

  /** For each airway generation, specify the angle mean
   *  to be used in the Gaussian-kernel density estimation. */
  void SetAngleMean( unsigned char, double );

  /** By default, all transitions from one state to another are set to 0.0
   *  (i.e. they are impossible). However, if the following method is called 
   *  for a given state transition, the specified means and variances will be
   *  used to compute the state transition using soure particle and target
   *  particle scales and orientations. The state transition probability is
   *  modeled as a product of Normal distributions: one for scale and one
   *  for the angular difference between the directions of the particles.
   *  Note that if this method is called for a given transition, it trumps
   *  anything set using 'SetTransitionProbability'. */
  void SetNormalTransitionProbabilityMeansAndVariances( unsigned char, unsigned char, 
							double, double, double, double );

  /** Set the transition probability between two states. This has no
   *  effect if the function 'SetNormalTransitionProbabilityMeansAndVariance'
   *  is called for the same transition */
  void SetTransitionProbability( unsigned char, unsigned char, double );

protected:
  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter();
  ~vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter();
 
  int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
 
private:
  //
  // In general, the particles may exist as a set of subgraphs.
  // We will need to run the HMM generation labeler independently
  // for each subgraph. This structure will facilitate keeping 
  // track of the various subgraphs.
  //
  struct SUBGRAPH
  {
    vtkSmartPointer< vtkMutableUndirectedGraph>  undirectedGraph;
    std::vector< vtkIdType >                     leafNodeIDs;
    std::map< vtkIdType, unsigned int >          nodeIDToParticleIDMap;
    std::map< unsigned int, vtkIdType >          particleIDToNodeIDMap;
  };

  //
  // The structures 'TRANSITIONPROBABILITYPARAMS' and 
  // 'TRANSITIONPROBABILITY' are used to facilitate
  // computation of transition probabilities. By default,
  // all transition probabilities are set to 0.0 (i.e. they are
  // all impossible). The user can set a constant probability for
  // a given transition, or the user can set Normal distribution
  // parameters (mean and variance) for a given transition. In
  // the latter case, the transition probability will be 
  // computed by looking at the scale difference between the 
  // source particle and target particle as well as by looking
  // at the angle between their direction vectors. Note that
  // if 'TRANSITIONPROBABILITYPARAMS' is set for a given 
  // transition, it will trump whatever is held by 
  // 'TRANSITIONPROBABILITY' for that transition.
  //
  struct TRANSITIONPROBABILITYPARAMETERS
  {
    unsigned char sourceState;
    unsigned char targetState;
    double        scaleDifferenceMean;
    double        scaleDifferenceVariance;
    double        angleMean;
    double        angleVariance;
  };

  struct TRANSITIONPROBABILITY
  {
    unsigned char sourceState;
    unsigned char targetState;
    double        probability;
  };

  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter(const vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter&);  // Not implemented.
  void operator=(const vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter&);  // Not implemented.

  double GetVectorMagnitude( double[3] );
  double GetAngleBetweenVectors( double[3], double[3], bool );
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
  void UpdateTrellisGraphWithViterbiStep( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph > );
  void BackTrack( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph >, std::map< unsigned int, unsigned char >*, double* );

  void ViewPolyData( vtkSmartPointer< vtkMutableUndirectedGraph > );
  void ViewGraph( vtkSmartPointer< vtkMutableUndirectedGraph > );
  void ViewGraph( vtkSmartPointer<vtkMutableDirectedGraph> );

  double EdgeWeightAngleSigma;

  unsigned int GetStateIndex( unsigned char );

  bool IsNonRootLeafNode( vtkIdType, vtkSmartPointer< vtkMutableDirectedGraph > );
  bool GetEdgeWeight( unsigned int, unsigned int, vtkSmartPointer< vtkPolyData >, double* );

  vtkSmartPointer< vtkMutableUndirectedGraph > MinimumSpanningTree;

  std::map< unsigned int, unsigned char >                    ParticleIDToAirwayGenerationMap;
  std::map< unsigned int, std::map<unsigned char, double> >  ParticleIDToEmissionProbabilitiesMap;
  std::map< float, double >                                  ScaleStandardDeviationsMap;
  std::map< float, double >                                  DistanceStandardDeviationsMap;
  std::map< float, double >                                  AngleStandardDeviationsMap;
  std::map< float, double >                                  ScaleMeansMap;
  std::map< float, double >                                  DistanceMeansMap;
  std::map< float, double >                                  AngleMeansMap;
  std::vector< SUBGRAPH >                                    Subgraphs;
  std::vector< unsigned char >                               States;
  std::vector< TRANSITIONPROBABILITYPARAMETERS >             NormalTransitionProbabilityParameters;
  std::vector< TRANSITIONPROBABILITY >                       TransitionProbabilities;
  std::vector< TRANSITIONPROBABILITY >                       BranchingTransitionProbabilities;
  std::vector< vtkSmartPointer< vtkPolyData > >              AirwayGenerationLabeledAtlases;

  double       ParticleDistanceThreshold;
  double       KernelDensityEstimationROIRadius;
  double       NoiseProbability;
  unsigned int NumberOfPointDataArrays;
  unsigned int NumberInputParticles;
  unsigned int NumberOfStates;
};
 
#endif
