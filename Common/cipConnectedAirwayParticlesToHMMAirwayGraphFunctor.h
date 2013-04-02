/**
 *  \class cipConnectedAirwayParticlesToHMMAirwayGraphFunctor
 *  \ingroup common
 *  \brief This functor builds a graph corresponding to 
 */

#ifndef __cipConnectedAirwayParticlesToHMMAirwayGraphFunctor_h
#define __cipConnectedAirwayParticlesToHMMAirwayGraphFunctor_h

#include "vtkPolyData.h"
#include "itkGraph.h"
#include "itkCIPHMMAirwayGraphTraits.h"
#include "vtkSmartPointer.h"

//
// TODO
// See notes in 'FilterAndLabelAirwayParticlesByGeneration.cxx' for
// ideas about KDE and transition probs.
// Make sure comments are consistent with current implementation
//
 
class cipConnectedAirwayParticlesToHMMAirwayGraphFunctor 
{
public:
  cipConnectedAirwayParticlesToHMMAirwayGraphFunctor();
  ~cipConnectedAirwayParticlesToHMMAirwayGraphFunctor();

  typedef itk::CIPHMMAirwayGraphTraits< double >  GraphTraitsType;
  typedef itk::Graph< GraphTraitsType >           GraphType;
  typedef GraphType::NodeIterator                 NodeIteratorType;
  typedef GraphType::NodeIdentifierType           NodeIdentifierType;
  typedef GraphType::NodePointerType              NodePointerType;
  typedef GraphType::EdgeIterator                 EdgeIteratorType;
  typedef GraphType::EdgeIdentifierType           EdgeIdentifierType;
  typedef GraphType::EdgePointerType              EdgePointerType;
  typedef GraphType::EdgeIdentifierContainerType  EdgeIdentifierContainerType;

  /** 
   *  The input is a VTK poly data corresponding to airway particles
   *  that have passed through the
   *  vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter (which
   *  reduces the number of noise particles but also imposes a
   *  topology on the particles indicating which particles should be
   *  connected to one another in the graph representation.
   */
  void SetInput( vtkSmartPointer< vtkPolyData > );

  /** 
   *  The Hidden Markov Model framework assumes we know the emission
   *  probabilities -- the conditional probabilities of the
   *  observations given the state of the latent variable (i.e. the
   *  airway generation label). This functor uses kernel density
   *  estimation to model the emission probabilities. The poly data
   *  passed to 'SetLabeledParticles' is assumed to be labeled (airway
   *  particle) training data, and it's assumed that the format
   *  conforms to the CIP guidelines for this type of data (TODO:
   *  create and reference the guidelines here). Training data derived
   *  from multiple training datasets / images should be packaged into
   *  one polydata, and all the training data should be properly
   *  registered to the coordinate frame of interest.
   */
  void SetLabeledParticles( vtkSmartPointer< vtkPolyData > );

  /**
   *  Kernel density estimation is used for the emission
   *  probabilities. Each state requires a collection of parameters to
   *  be used for the kernel density estimation. For the following
   *  function, the first argument specifies the state (airway
   *  generation, etc). The following parameters are as follows:
   *  distance lambda, angle lambda, and scale difference sigma. A
   *  zero-mean Gaussian kernel is used for the scale difference,
   *  and exponential kernels are used for distance and angle. This
   *  function needs to be called for each state. (Note that in the
   *  current implementation, it's assumed that this function is
   *  called exactly once for each state. This relates to establishing
   *  a unique list of states and a correct number of states).
   */
  void SetStateKernelDensityEstimatorParameters( unsigned char, double, double, double );

  /**
   *  Transition matrices are computed individually for each
   *  node-to-node connection based on relative scale difference and
   *  the angle formed between the vector connecting the particles and
   *  the target node's direction. The parameters are as follows: from
   *  lung type, to lung type, angle lambda, relative scale mu,
   *  relative scale sigma. The state transition probability from the
   *  'from lung type' to the 'to lung type' is modeled as the product
   *  of a Gaussian and an exponential distribution. Every possible
   *  (non-zero) transition should be indicated with a call to this
   *  function with the appropriate parameters. If a transition from
   *  one state to another is not specifically designated with a call
   *  to this function, that transition will be assumed to have 0
   *  probability.
   */
  void SetStateTransitionParameters( unsigned char, unsigned char, double, double, double );
  
  /**
   *  The output is an itkGraph that can be passed to the
   *  'itkCIPHMMAirwayGraphToStateLabeledAirwayGraphFilter' so that
   *  the optimal generation labels for each particle can be assigned
   *  by finding the min-cost path through the graphs.
   */
  //GraphType::Pointer GetOutput();

  void Update();

protected:
  
private:
  /**
   *  KDE stands for kernel density estimation. This structure is a
   *  container for the different parameters that are used for
   *  KDE-based construction of the emission probabilities.
   */
  struct KDEPARAMS
    {
      double distanceLambda;
      double angleLambda;
      double scaleDifferenceSigma;
    };

  /**
   *  This structure contains parameters needed for computing
   *  transition probabilities. 
   */
  struct TRANSITIONPARAMS
    {
      unsigned char fromLungType;
      unsigned char toLungType;
      double angleLambda;
      double relativeScaleMu;
      double relativeScaleSigma;
    };

  void   EstablishSequenceOrdering( NodePointerType );
  void   ComputeEmissionProbabilities();
  void   ComputeTransitionProbabilityMatrices();
  void   ComputeTransitionProbabilityMatrix( EdgeIdentifierType, double, double );
  bool   GetParticleExistsInGraph( unsigned int );
  void   InitializeRoot();
  void   CreateEdgesBetweenParticles( unsigned int, unsigned int );
  double GetAngleBetweenVectors( double[3], double[3], bool );
  double GetVectorMagnitude( double[3] );
  double GetEmissionProbabilityContribution( unsigned char, double, double, double );

  std::map< unsigned char, KDEPARAMS >     StateKDEParameters;
  std::map< unsigned char, unsigned int >  NumberOfLabeledParticlesInState;
  std::vector< TRANSITIONPARAMS >          StateTransitionParameters;
  std::vector< unsigned char >             States;
  vtkSmartPointer< vtkPolyData >           InputParticlesData;
  vtkSmartPointer< vtkPolyData >           LabeledParticlesData;
  GraphType::Pointer                       OutputGraph;
  unsigned int                             NumberOfStates;

  unsigned int RootParticleID;
};
 
#endif
