#include "cipConnectedAirwayParticlesToHMMAirwayGraphFunctor.h"

#include "vtkGenericCell.h"
#include "vtkFieldData.h"
#include <cfloat>
#include <math.h>

//
// Note: PROJ identifier used to indicate project specific code
// elements 
//

//
// TODO
// 'LungRegion' has been used throughout to identify the float array
// that contains the labeled information. This is because the data I
// originally created was created incorrectly. Should be 'LungType',
// and the test data should be changed to reflect this.
//
// I think the graph to poly vtk filter makes redundancies. That's why
// we have this 'ParticleExistsInGraph' function.
//

cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::cipConnectedAirwayParticlesToHMMAirwayGraphFunctor()
{
  this->InputParticlesData    = vtkSmartPointer< vtkPolyData >::New();
  this->LabeledParticlesData  = vtkSmartPointer< vtkPolyData >::New();
  this->OutputGraph           = GraphType::New();
  this->NumberOfStates        = 0;
}
 

cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::~cipConnectedAirwayParticlesToHMMAirwayGraphFunctor()
{ 
}

void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::SetInput( vtkSmartPointer< vtkPolyData > particles )
{
  this->InputParticlesData = particles;
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::SetLabeledParticles( vtkSmartPointer< vtkPolyData > particles )
{
  this->LabeledParticlesData = particles;
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::Update()
{
  //
  // In general, there will be several subgraphs in
  // 'InputParticlesData'. We need to create trellis graphs for
  // each. Begin by creating a graph that has one node associated with
  // each particle, creating bi-directional links between linked
  // particles. Once this graph has been created, we'll need to
  // identify leaf nodes and subsequently identify parent nodes for
  // each subGraph using Murray's Law. Once parent nodes have been
  // established, we can expand the graph as a trellis graph by adding
  // N+1 nodes for each existing node (where N is the number of airway
  // generations we're attempting to label, and adding an additional
  // node to account for possible noise).
  //

  //
  // The cells produced by the
  // vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.h filter
  // have two points each, forming a small line segment. For every
  // cell, we will add an edge between the two points, and we'll check
  // for connections to other, previously added points and add
  // connections as necessary.
  //

  vtkSmartPointer< vtkGenericCell > cell = vtkSmartPointer< vtkGenericCell >::New();

  unsigned int particleID1, particleID2;

  for ( unsigned int i=0; i<this->InputParticlesData->GetNumberOfCells(); i++ )
    {
    this->InputParticlesData->GetCell( i, cell );
    
    particleID1 = cell->GetPointId( 0 );
    particleID2 = cell->GetPointId( 1 );

    //
    // Loop through current state of graph. If either of these two
    // particles is not represented in the graph, create nodes for them
    //
    if ( !this->GetParticleExistsInGraph( particleID1 ) )
      {
      NodePointerType nodePtr = this->OutputGraph->CreateNewNode();
      nodePtr->ParticleID = particleID1;
      }
    if ( !this->GetParticleExistsInGraph( particleID2 ) )
      {
      NodePointerType nodePtr = this->OutputGraph->CreateNewNode();
      nodePtr->ParticleID = particleID2;
      }
    
    //
    // Now create edges between these two particles.
    //
    this->CreateEdgesBetweenParticles( particleID1, particleID2 );
    }
    
  std::cout << "this->OutputGraph->GetTotalNumberOfNodes():\t" << this->OutputGraph->GetTotalNumberOfNodes() << std::endl;
  std::cout << "this->OutputGraph->GetTotalNumberOfEdges():\t" << this->OutputGraph->GetTotalNumberOfEdges() << std::endl;

  //
  // At this stage, we've created the bidirectional graph and have
  // identified the root node (PROJ). We now want to establish
  // direction of flow from leaves to root. Direction of flow will be
  // designated by setting the weights of edges indicating proper flow
  // direction to 0, and 1 otherwise.
  //
  // Note that we'll probably want to put the root identification and
  // direction of flow identification into separate routines 
  //
  // TODO -- 'InitialzeRoot' needs to be generalized. as of 12/9/11 it
  // naively picks the particle with the largest z value (assuming
  // trachea). This is sufficient for time being for testing other
  // components on simple training set I created (see notes in
  // ~/NEU/MachineLearning. Notes also in
  // ~/Conferences/TMI/AirwayGenerationLabeling.
  //
  this->InitializeRoot();

  NodeIteratorType nIt( this->OutputGraph );


  //
  // TODO
  // The 'EstablishSequenceOrdering' routine will need to be extended
  // to handle the case of some random subtree. Right now it's just
  // implemented for our test data -- labeled gens 0-2.
  //
  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() ) 
    {
    if (  nIt.Get().ParticleID == this->RootParticleID )
      {
      this->EstablishSequenceOrdering( nIt.GetPointer() );
      }

    ++nIt;
    }

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() ) 
    {
    EdgeIdentifierContainerType outEdges = this->OutputGraph->GetOutgoingEdges( nIt.GetPointer() );
    
    unsigned int fromParticleID = nIt.Get().ParticleID;

    double fromLungType = this->InputParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( fromParticleID )[0];

    double fromPoint[3];
      fromPoint[0] = this->InputParticlesData->GetPoint( fromParticleID )[0];
      fromPoint[1] = this->InputParticlesData->GetPoint( fromParticleID )[1];
      fromPoint[2] = this->InputParticlesData->GetPoint( fromParticleID )[2];

    double fromScale = this->InputParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( fromParticleID )[0];

    double fromHevec2[3];
      fromHevec2[0] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[0];
      fromHevec2[1] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[1];
      fromHevec2[2] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[2];

    for ( unsigned int i=0; i<outEdges.size(); i++ )
      {
      unsigned int toParticleID = this->OutputGraph->GetTargetNode( outEdges[i] ).ParticleID;

      if ( this->OutputGraph->GetEdgeWeight( outEdges[i] ) == 0 )
        {
        double toLungType = this->InputParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( toParticleID )[0];

        double toPoint[3];
          toPoint[0] = this->InputParticlesData->GetPoint( toParticleID )[0];
          toPoint[1] = this->InputParticlesData->GetPoint( toParticleID )[1];
          toPoint[2] = this->InputParticlesData->GetPoint( toParticleID )[2];
        
        double toScale = this->InputParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( toParticleID )[0];

        double toHevec2[3];
          toHevec2[0] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[0];
          toHevec2[1] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[1];
          toHevec2[2] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[2];

        double connectingVec[3];
          connectingVec[0] = toPoint[0] - fromPoint[0];
          connectingVec[1] = toPoint[1] - fromPoint[1];
          connectingVec[2] = toPoint[2] - fromPoint[2];

//         std::cout << fromLungType << "\t" << toLungType << "\t";
//         std::cout << (toScale-fromScale)/fromScale << "\t";
//          std::cout << "angle:\t" << this->GetAngleBetweenVectors( connectingVec, toHevec2, true ) << std::endl;
//         std::cout << this->GetVectorMagnitude( connectingVec ) << std::endl;
        }
      }

    ++nIt;
    }

  //
  // Compute emission probabilities for all particles
  //
  this->ComputeEmissionProbabilities();
  this->ComputeTransitionProbabilityMatrices();
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::SetStateTransitionParameters( unsigned char fromLungType, unsigned char toLungType, 
                                                                                       double angleLambda, double relativeScaleMu, double relativeScaleSigma )
{
  TRANSITIONPARAMS params;
    params.fromLungType       = fromLungType;
    params.toLungType         = toLungType;
    params.angleLambda        = angleLambda;
    params.relativeScaleMu    = relativeScaleMu;
    params.relativeScaleSigma = relativeScaleSigma;

  this->StateTransitionParameters.push_back( params );
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::SetStateKernelDensityEstimatorParameters( unsigned char lungType, double distanceLambda,
                                                                                                   double angleLambda, double scaleDifferenceSigma )
{
  KDEPARAMS params;
    params.distanceLambda       = distanceLambda;
    params.angleLambda          = angleLambda;
    params.scaleDifferenceSigma = scaleDifferenceSigma;

  this->StateKDEParams[lungType] = params;

  this->States.push_back( lungType );
  this->NumberOfStates = this->States.size();
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::ComputeTransitionProbabilityMatrices()
{
  unsigned int fromParticleID, toParticleID;

  NodeIteratorType nIt( this->OutputGraph );

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    fromParticleID = nIt.Get().ParticleID;

    double fromPoint[3];
      fromPoint[0] = this->InputParticlesData->GetPoint( fromParticleID )[0];
      fromPoint[1] = this->InputParticlesData->GetPoint( fromParticleID )[1];
      fromPoint[2] = this->InputParticlesData->GetPoint( fromParticleID )[2];

    double fromHevec2[3];
      fromHevec2[0] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[0];
      fromHevec2[1] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[1];
      fromHevec2[2] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( fromParticleID )[2];

    double fromScale = this->InputParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( fromParticleID )[0];

    EdgeIdentifierContainerType outEdges = this->OutputGraph->GetOutgoingEdges( nIt.GetPointer() );

    for ( unsigned int i=0; i<outEdges.size(); i++ )
      {
      if ( this->OutputGraph->GetEdgeWeight( outEdges[i] ) == 0 )
        {
        toParticleID = this->OutputGraph->GetTargetNode( outEdges[i] ).ParticleID;

        double toPoint[3];
          toPoint[0] = this->InputParticlesData->GetPoint( toParticleID )[0];
          toPoint[1] = this->InputParticlesData->GetPoint( toParticleID )[1];
          toPoint[2] = this->InputParticlesData->GetPoint( toParticleID )[2];

        double toHevec2[3];
          toHevec2[0] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[0];
          toHevec2[1] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[1];
          toHevec2[2] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( toParticleID )[2];

        double connectingVec[3];
          connectingVec[0] = toPoint[0] - fromPoint[0];
          connectingVec[1] = toPoint[1] - fromPoint[1];
          connectingVec[2] = toPoint[2] - fromPoint[2];

        double angle = this->GetAngleBetweenVectors( connectingVec, toHevec2, true );

        double toScale = this->InputParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( toParticleID )[0];

        double relativeScale = (toScale-fromScale)/fromScale;

        this->ComputeTransitionProbabilityMatrix( outEdges[i], angle, relativeScale );
        }
      }

    ++nIt;
    }
}


// TODO
// How to incorporate Murray's Law? Use it at all? Just use data
// driven approach? Kernels to use? Gaussian? Exponential? Log normal?
void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::ComputeTransitionProbabilityMatrix( EdgeIdentifierType ePtr, double angle, double relativeScale )
{
  double pi = 3.14159265358979323846;

  unsigned int fromParticleID = this->OutputGraph->GetSourceNode( ePtr ).ParticleID;
  unsigned int toParticleID   = this->OutputGraph->GetTargetNode( ePtr ).ParticleID;
  double fromLungType = this->InputParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( fromParticleID )[0];
  double toLungType = this->InputParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( toParticleID )[0];
  std::cout << "------------------------------" << std::endl;
  std::cout << fromLungType << "\t" << toLungType << "\t" << angle << std::endl;
  std::cout << "angle, relativeScale:\t" << angle << "\t" << relativeScale << std::endl;

  //
  // Initialize this edge's transition matrix
  //
  for ( unsigned int r=0; r<this->NumberOfStates; r++ )
    {
    std::vector< double > temp;
    for ( unsigned int c=0; c<this->NumberOfStates; c++ )
      {
      temp.push_back( 0 );
      }
    this->OutputGraph->GetEdge( ePtr ).TransitionMatrix.push_back( temp );
    }

  double lambda, mu, sig;
  double scaleTerm, angleTerm;
  unsigned int row, col;
  for ( unsigned int i=0; i<this->StateTransitionParameters.size(); i++ )
    {
    lambda = this->StateTransitionParameters[i].angleLambda;
    mu     = this->StateTransitionParameters[i].relativeScaleMu;
    sig    = this->StateTransitionParameters[i].relativeScaleSigma;

    scaleTerm = (1.0/(sqrt(2.0*pi)*sig))*exp(-pow( relativeScale-mu, 2 )/(2.0*sig*sig));
    angleTerm = lambda*exp(-lambda*angle);

    std::cout << "scaleTerm, angleTerm:\t" << scaleTerm << "\t" << angleTerm << std::endl;

    for ( unsigned int j=0; j<this->NumberOfStates; j++ )
      {
      if ( this->StateTransitionParameters[i].fromLungType == this->States[j] )
        {
        row = j;
        }
      if ( this->StateTransitionParameters[i].toLungType == this->States[j] )
        {
        col = j;
        }
      }

    this->OutputGraph->GetEdge( ePtr ).TransitionMatrix[row][col] = scaleTerm*angleTerm;
    }

  //
  // Now normalize the rows of the matrix
  //
  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
    double mag = 0.0;
    for ( unsigned int j=0; j<this->NumberOfStates; j++ )
      {
      mag += pow( this->OutputGraph->GetEdge( ePtr ).TransitionMatrix[i][j], 2);
      }
    mag = sqrt( mag );
    std::cout << "mag:\t" << mag << std::endl;
    for ( unsigned int j=0; j<this->NumberOfStates; j++ )
      {
      this->OutputGraph->GetEdge( ePtr ).TransitionMatrix[i][j] /= mag;
      std::cout << this->OutputGraph->GetEdge( ePtr ).TransitionMatrix[i][j] << "\t";
      }
    std::cout << std::endl;
    }

}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::ComputeEmissionProbabilities()
{
  NodeIteratorType nIt( this->OutputGraph );

  double position[3], positionLabeled[3];
  double scale, scaleLabeled;
  double hevec2[3], hevec2Labeled[3];
  double connectingVec[3];
  double distance, angle, scaleDifference;

  unsigned int  particleID;
  unsigned char lungType;

  unsigned int correct = 0;
  unsigned int total   = 0;

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() ) 
    {
    particleID = nIt.Get().ParticleID;

    scale = this->InputParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( particleID )[0];

    hevec2[0] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( particleID )[0];
    hevec2[1] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( particleID )[1];
    hevec2[2] = this->InputParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( particleID )[2];
      
    position[0] = this->InputParticlesData->GetPoint( particleID )[0];
    position[1] = this->InputParticlesData->GetPoint( particleID )[1];
    position[2] = this->InputParticlesData->GetPoint( particleID )[2];

    std::map< unsigned char, double >        emissionProbabilities;
    std::map< unsigned char, unsigned int >  statesCounter;

    for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      {
      emissionProbabilities[this->States[i]] = 0;
      statesCounter[this->States[i]] = 0;
      }

    for ( unsigned int i=0; i<this->LabeledParticlesData->GetNumberOfPoints(); i++ )
      {
      lungType = static_cast< unsigned char >( this->LabeledParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( i )[0] );

      scaleLabeled = this->LabeledParticlesData->GetFieldData()->GetArray( "scale" )->GetTuple( i )[0];

      hevec2Labeled[0] = this->LabeledParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( i )[0];
      hevec2Labeled[1] = this->LabeledParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( i )[1];
      hevec2Labeled[2] = this->LabeledParticlesData->GetFieldData()->GetArray( "hevec2" )->GetTuple( i )[2];
      
      positionLabeled[0] = this->LabeledParticlesData->GetPoint( i )[0];
      positionLabeled[1] = this->LabeledParticlesData->GetPoint( i )[1];
      positionLabeled[2] = this->LabeledParticlesData->GetPoint( i )[2];

      connectingVec[0] = position[0]-positionLabeled[0];
      connectingVec[1] = position[1]-positionLabeled[1];
      connectingVec[2] = position[2]-positionLabeled[2];

      distance = this->GetVectorMagnitude( connectingVec );
      angle = this->GetAngleBetweenVectors( hevec2, hevec2Labeled, true );
      scaleDifference = scale - scaleLabeled;

      //
      // TODO
      // At last minute, tried to just use an epsilon ball around the
      // particle for the kernel density estimation. Is this
      // appropriate?
      //
      if ( distance < 20.0 )
        {
        emissionProbabilities[lungType] += this->GetEmissionProbabilityContribution( lungType, distance, angle, scaleDifference );
        statesCounter[lungType]++;
        }
      }

//     double max = 0;     //PROJ
//     unsigned char bestGuess;     //PROJ

    //   double mag = 0;
    for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      {
      if ( statesCounter[this->States[i]] == 0 )
        {
        emissionProbabilities[this->States[i]] = 0.0;
        }
      else
        {
        emissionProbabilities[this->States[i]] /= static_cast< double >( statesCounter[this->States[i]] );
//        mag += pow( emissionProbabilities[this->States[i]], 2 );
        }
      }
//     mag = sqrt( mag );
//     for ( unsigned int i=0; i<this->NumberOfStates; i++ )
//       {
//       emissionProbabilities[this->States[i]] /= mag;

//       nIt.Get().EmissionProbability[this->States[i]] = emissionProbabilities[this->States[i]];

// //       if ( emissionProbabilities[this->States[i]] > max )
// //         {
// //         max = emissionProbabilities[this->States[i]];
// //         bestGuess = this->States[i];
// //         }
//       }

//     std::cout << this->LabeledParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( nIt.Get().ParticleID )[0] << "\t";
//     std::cout << static_cast< double >( bestGuess ) << "\t" << max << std::endl;
//     if ( this->LabeledParticlesData->GetFieldData()->GetArray( "LungRegion" )->GetTuple( nIt.Get().ParticleID )[0] == static_cast< double >( bestGuess ) )
//       {
//       correct++;
//       }
//     total++;

    ++nIt;
    }
  
//  std::cout << static_cast< double >(correct)/static_cast< double >(total) << std::endl;
}


double cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::GetEmissionProbabilityContribution( unsigned char lungType, double distance, 
                                                                                               double angle, double scaleDifference )
{
  double contribution;

  double lambda, sig;
  double pi = 3.14159265358979323846;

  //
  // Distance term
  //
  lambda  = this->StateKDEParams[lungType].distanceLambda;
  
  double distanceContribution = lambda*exp(-lambda*distance);

  //
  // Angle term
  //
  lambda  = this->StateKDEParams[lungType].angleLambda;
  
  double angleContribution = lambda*exp(-lambda*angle);

  //
  // Scale difference term
  //
  sig = this->StateKDEParams[lungType].scaleDifferenceSigma;
  
  double scaleDifferenceContribution = (1.0/(sqrt(2.0*pi)*sig))*exp(-pow( scaleDifference, 2 )/(2.0*sig*sig));

  contribution = distanceContribution*angleContribution*scaleDifferenceContribution;

  return contribution;
}


bool cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::GetParticleExistsInGraph( unsigned int particleID )
{
  double inPoint[3];
  inPoint[0] = this->InputParticlesData->GetPoint( particleID )[0];
  inPoint[1] = this->InputParticlesData->GetPoint( particleID )[1];
  inPoint[2] = this->InputParticlesData->GetPoint( particleID )[2];

  double loopPoint[3];

  NodeIteratorType nIt( this->OutputGraph );

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    loopPoint[0] = this->InputParticlesData->GetPoint( nIt.Get().ParticleID )[0];
    loopPoint[1] = this->InputParticlesData->GetPoint( nIt.Get().ParticleID )[1];
    loopPoint[2] = this->InputParticlesData->GetPoint( nIt.Get().ParticleID )[2];
    
    if ( loopPoint[0] == inPoint[0] && loopPoint[1] == inPoint[1] && loopPoint[2] == inPoint[2] )
      {
      return true;
      }

    ++nIt;
    }

  return false;
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::EstablishSequenceOrdering( NodePointerType nPtr )
{
  //
  // The convention will be that an edge weight of 0 corresponds to
  // the correct direction of flow ('0' for "no resistance")
  //
  EdgeIdentifierContainerType outEdges = this->OutputGraph->GetOutgoingEdges( nPtr );
  EdgeIdentifierContainerType inEdges  = this->OutputGraph->GetIncomingEdges( nPtr );

  for ( unsigned int i=0; i<outEdges.size(); i++ )
    {
    if ( this->OutputGraph->GetEdgeWeight( inEdges[i] ) == 2.0 )
      {
      this->OutputGraph->SetEdgeWeight( inEdges[i], 0.0 );
      }
    if ( this->OutputGraph->GetEdgeWeight( outEdges[i] ) == 2.0 )
      {
      this->OutputGraph->SetEdgeWeight( outEdges[i], 1.0 );
      this->EstablishSequenceOrdering( &this->OutputGraph->GetTargetNode( outEdges[i] ) );
      }
    }      
}


void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::CreateEdgesBetweenParticles( unsigned int particleID1, unsigned int particleID2 )
{
  double point1[3];
    point1[0] = this->InputParticlesData->GetPoint( particleID1 )[0];
    point1[1] = this->InputParticlesData->GetPoint( particleID1 )[1];
    point1[2] = this->InputParticlesData->GetPoint( particleID1 )[2];

  double point2[3];
    point2[0] = this->InputParticlesData->GetPoint( particleID2 )[0];
    point2[1] = this->InputParticlesData->GetPoint( particleID2 )[1];
    point2[2] = this->InputParticlesData->GetPoint( particleID2 )[2];

  double loopPoint[3];

  NodeIdentifierType nodeIdentifier1, nodeIdentifier2;

  NodeIteratorType nIt( this->OutputGraph );

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    unsigned int loopParticleID = nIt.Get().ParticleID; 

    loopPoint[0] = this->InputParticlesData->GetPoint( loopParticleID )[0];
    loopPoint[1] = this->InputParticlesData->GetPoint( loopParticleID )[1];
    loopPoint[2] = this->InputParticlesData->GetPoint( loopParticleID )[2];

    if ( point1[0] == loopPoint[0] && point1[1] == loopPoint[1] && point1[2] == loopPoint[2] )
      {
      nodeIdentifier1 = nIt.Get().Identifier;
      }
    if ( point2[0] == loopPoint[0] && point2[1] == loopPoint[1] && point2[2] == loopPoint[2] )
      {
      nodeIdentifier2 = nIt.Get().Identifier;
      }

    ++nIt;
    }

  bool addEdges = true;

  NodePointerType nodePtr = this->OutputGraph->GetNodePointer( nodeIdentifier1 );
  EdgeIdentifierContainerType outEdges = this->OutputGraph->GetOutgoingEdges( nodePtr );
  
  NodePointerType targetPtr;

  for ( unsigned int i=0; i<outEdges.size(); i++ )
    {
    targetPtr = this->OutputGraph->GetTargetNodePointer( outEdges[i] ); 
    if ( this->OutputGraph->GetNodeIdentifier( targetPtr ) == nodeIdentifier2 )
      {
      addEdges = false;
      break;
      }
    }

  if ( addEdges )
    {
    EdgePointerType edgePtr;
    edgePtr = this->OutputGraph->CreateNewEdge( nodeIdentifier1, nodeIdentifier2 );
    this->OutputGraph->SetEdgeWeight( edgePtr, 2.0 );

    edgePtr = this->OutputGraph->CreateNewEdge( nodeIdentifier2, nodeIdentifier1 );
    this->OutputGraph->SetEdgeWeight( edgePtr, 2.0 );
    }
}


//PROJ
void cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::InitializeRoot()
{
  double z;  
  double maxZ = -DBL_MAX; 

  NodeIteratorType nIt( this->OutputGraph );

  nIt.GoToBegin();
  while ( !nIt.IsAtEnd() )
    {
    z = this->InputParticlesData->GetPoint( nIt.Get().ParticleID )[2];
    
    if ( z > maxZ )
      {
      maxZ = z;
      this->RootParticleID = nIt.Get().ParticleID;
      }

    ++nIt;
    }
}
 

double cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::GetVectorMagnitude( double vector[3] )
{
  double magnitude = sqrt( pow( vector[0], 2 ) + pow( vector[1], 2 ) + pow( vector[2], 2 ) );

  return magnitude;
}


double cipConnectedAirwayParticlesToHMMAirwayGraphFunctor::GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees )
{
  double vec1Mag = this->GetVectorMagnitude( vec1 );
  double vec2Mag = this->GetVectorMagnitude( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = acos( arg );

  if ( !returnDegrees )
    {
    return angle;
    }

  double angleInDegrees = (180.0/3.14159265358979323846)*angle;

  if ( angleInDegrees > 90.0 )
    {
    angleInDegrees = 180.0 - angleInDegrees;
    }

  return angleInDegrees;
}


//AddStateNodes
//GetEmissionProbabilities( position, direction, scale, std::map<
//double >* probabilities )
