/**
 *
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 */

#include "vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.h"

#include "vtkGlyphSource2D.h"
#include "vtkSimple2DLayoutStrategy.h"
#include "cipConventions.h"
#include "vtkMath.h"
#include "vtkIdTypeArray.h"
#include "vtkSelectionNode.h"
#include "vtkPolyDataWriter.h"
#include "vtkIndent.h"
#include "vtkExtractSelectedGraph.h"
#include "vtkSelection.h"
#include "vtkProperty.h" 
#include "vtkDataSetAttributes.h"
#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkDataObject.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkGraphLayoutView.h"
#include "vtkGraphLayout.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#include "vtkVertexListIterator.h"
#include "vtkActor.h"
#include "vtkGraphToPolyData.h"
#include "vtkSphereSource.h"
#include "vtkGlyph3D.h"
#include "vtkMutableUndirectedGraph.h"
#include "vtkBoostConnectedComponents.h"
#include "vtkUnsignedIntArray.h"
#include "vtkOutEdgeIterator.h"
#include "vtkEdgeListIterator.h"
#include "vtkInEdgeIterator.h"
#include <cfloat>
#include <math.h>
#include <list>


vtkStandardNewMacro( vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter );
 
vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter()
{
  this->EdgeWeightAngleSigma                    = 1.0;
  this->NoiseProbability                        = DBL_MIN; //TODO: How best to handle noise?
  this->KernelDensityEstimationROIRadius        = DBL_MAX;
  this->NumberOfStates                          = 10;//12; //11 airway generations, 1 noise

  //  this->States.push_back( static_cast< unsigned char >( cip::UNDEFINEDTYPE ) );
  this->States.push_back( static_cast< unsigned char >( cip::TRACHEA ) );
  this->States.push_back( static_cast< unsigned char >( cip::MAINBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION6 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION7 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION8 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION9 ) );
  // this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION10 ) );

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
      for ( unsigned int j=0; j<this->NumberOfStates; j++ )
	{
	  TRANSITIONPROBABILITY transProb;
	  transProb.sourceState = this->States[i];
	  transProb.targetState = this->States[j];

	  if ( (this->States[i] >= this->States[j]) && this->States[i] != (unsigned char)( cip::UNDEFINEDTYPE ) && i-j <= 2 )
	    {
	      transProb.probability = 1e-20;
	    }
	  else if ( this->States[i] == (unsigned char)( cip::UNDEFINEDTYPE ) ||
		    this->States[j] == (unsigned char)( cip::UNDEFINEDTYPE ) )
	    {
	      transProb.probability = 1e-20;
	    }
	  else
	    {
	      transProb.probability = 0.0;
	    }
	  
	  this->TransitionProbabilities.push_back( transProb );
	  this->BranchingTransitionProbabilities.push_back( transProb );
	}
    }

  //
  // Initialize the branching transition probabilities. These are probabilities
  // that are used whenever a branch point is detected.
  //
  for ( unsigned int i=0; i<this->BranchingTransitionProbabilities.size(); i++ )
    {
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::MAINBRONCHUS) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::TRACHEA) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.9999999;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::MAINBRONCHUS) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.0000001;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.9999999;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.0000001;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::TRACHEA) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.0;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION3) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.2;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION4) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION4) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION3) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION4) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION4) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION5) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION3) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION5) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION4) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION5) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION5) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION6) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION4) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION6) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION5) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION6) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION6) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION7) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION5) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION7) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION6) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION7) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION7) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION8) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION6) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION8) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION7) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.8;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION8) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION8) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION9) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION8) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.9;
	}
      if ( this->BranchingTransitionProbabilities[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION9) &&
	   this->BranchingTransitionProbabilities[i].targetState == (unsigned char)(cip::AIRWAYGENERATION9) )
	{
	  this->BranchingTransitionProbabilities[i].probability = 0.1;
	}
    }

  //
  // Note that we loop from i=1 because noise (cip::UNDEFINTEDTYPE) is treated
  // as having a uniform distribution
  //
  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
      this->ScaleStandardDeviationsMap[static_cast< float >( this->States[i] )]     = 1.0;
      this->DistanceStandardDeviationsMap[static_cast< float >( this->States[i] )]  = 1.0;
      this->AngleStandardDeviationsMap[static_cast< float >( this->States[i] )]     = 1.0;
      this->ScaleMeansMap[static_cast< float >( this->States[i] )]                  = 0.0;
      this->DistanceMeansMap[static_cast< float >( this->States[i] )]               = 0.0;
      this->AngleMeansMap[static_cast< float >( this->States[i] )]                  = 0.0;
    }

  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}


vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::~vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter()
{
  
}
 

void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetTransitionProbability( unsigned char sourceState, unsigned char targetState, 
											      double probability )
{
  bool found = false;
  for ( unsigned int i=0; i<this->TransitionProbabilities.size(); i++ )
    {
      if ( this->TransitionProbabilities[i].sourceState == sourceState &&
	   this->TransitionProbabilities[i].targetState == targetState )
	{
	  this->TransitionProbabilities[i].probability = probability;
	  
	  found = true;
	}
    }
  
  if ( !found )
    {
      TRANSITIONPROBABILITY transProb;
        transProb.sourceState = sourceState;
	transProb.targetState = targetState;
	transProb.probability = probability;

      this->TransitionProbabilities.push_back( transProb );
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetNormalTransitionProbabilityMeansAndVariances( unsigned char sourceState, unsigned char targetState, 
														     double scaleDiffMean, double scaleDiffVariance,
														     double angleMean, double angleVariance )
{
  cip::ChestConventions conventionsDEB;
  std::cout << "---------------------------" << std::endl;
  std::cout << "Setting:\t" << conventionsDEB.GetChestTypeName( sourceState ) << "\t to:\t";
  std::cout << conventionsDEB.GetChestTypeName( targetState ) << std::endl;

  bool found = false;
  for ( unsigned int i=0; i<this->NormalTransitionProbabilityParameters.size(); i++ )
    {
      if ( this->NormalTransitionProbabilityParameters[i].sourceState == sourceState &&
	   this->NormalTransitionProbabilityParameters[i].targetState == targetState )
	{
	  std::cout << "Found" << std::endl;
	  this->NormalTransitionProbabilityParameters[i].scaleDifferenceMean     = scaleDiffMean;
	  this->NormalTransitionProbabilityParameters[i].scaleDifferenceVariance = scaleDiffVariance;
	  this->NormalTransitionProbabilityParameters[i].angleMean               = angleMean;
	  this->NormalTransitionProbabilityParameters[i].angleVariance           = angleVariance;

	  std::cout << this->NormalTransitionProbabilityParameters[i].scaleDifferenceMean << std::endl;
	  std::cout << this->NormalTransitionProbabilityParameters[i].scaleDifferenceVariance << std::endl;
	  std::cout << this->NormalTransitionProbabilityParameters[i].angleMean << std::endl;
	  std::cout << this->NormalTransitionProbabilityParameters[i].angleVariance << std::endl;
	  
	  found = true;
	}
    }
  
  if ( !found )
    {
      std::cout << "Not found" << std::endl;

      TRANSITIONPROBABILITYPARAMETERS params;
        params.sourceState              = sourceState;
	params.targetState              = targetState;
	params.scaleDifferenceMean      = scaleDiffMean;
	params.scaleDifferenceVariance  = scaleDiffVariance;
	params.angleMean                = angleMean;
	params.angleVariance            = angleVariance;

	std::cout << params.scaleDifferenceMean << std::endl;
	std::cout << params.scaleDifferenceVariance << std::endl;
	std::cout << params.angleMean << std::endl;
	std::cout << params.angleVariance << std::endl;

      this->NormalTransitionProbabilityParameters.push_back( params );
    }
}

 
int vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::RequestData(vtkInformation *vtkNotUsed(request),
									       vtkInformationVector **inputVector,
									       vtkInformationVector *outputVector)
{
  vtkInformation* inInfo  = inputVector[0]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
 
  vtkSmartPointer< vtkPolyData > inputParticles  = vtkPolyData::SafeDownCast( inInfo->Get(vtkDataObject::DATA_OBJECT()) );
  vtkSmartPointer< vtkPolyData > outputParticles = vtkPolyData::SafeDownCast( outInfo->Get(vtkDataObject::DATA_OBJECT()) );

  this->NumberInputParticles    = inputParticles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = inputParticles->GetPointData()->GetNumberOfArrays();

  //
  // Initialized the airway generation assignments. Each particle
  // will be assigned cip::UNDEFINEDTYPE to initialize. Calling
  // this function will fill 'this->ParticleIDToAirwayGenerationMap'
  // for subsequent use.
  //
  this->InitializeAirwayGenerationAssignments( this->NumberInputParticles );
  
  //
  // Compute the emission probabilities. We only need to do this once. 
  // The emission probabilites are independent of the graph structure
  // of the particles. Calling this function will fill out
  // 'this->ParticleIDToEmissionProbabilitiesMap' for subsequent use.
  // 
  std::cout << "---Initializing emission probs..." << std::endl;
  this->InitializeEmissionProbabilites( inputParticles );
  std::cout << "---DONE." << std::endl;  

  //
  // Compute the minimum spanning tree that will be used to establish
  // directionality through the particles. Calling this function will 
  // fill out 'this->MinimumSpanningTree' for subsequent use
  // 
  std::cout << "---Initializing minimum spanning tree..." << std::endl;
  this->InitializeMinimumSpanningTree( inputParticles );
  std::cout << "---DONE." << std::endl;
  // //this->ViewPolyData( this->MinimumSpanningTree );

  //
  // Now that we have initialized the minimum spanning tree, we can
  // construct the subgraphs (in general it make be the case that
  // several subgraphs / subtrees exist. We will want to comptue the 
  // HMM labeler for all of them). Calling this function will fill
  // out 'this->Subgraphs' for subsequent use.
  //
  std::cout << "---Initializing subgraphs..." << std::endl;
  this->InitializeSubGraphs( this->MinimumSpanningTree, inputParticles );
  std::cout << "---DONE." << std::endl;

  // 
  // Now for each subgraph, consider each leaf node in turn, assume
  // it is the root node, and perform the labeling. Whichever leaf node
  // acts as the most probable root node (based on the greatest likelihood)
  // will be considered the true root node for that subtree, and the 
  // corresponding generation labels will be used.
  // 
  std::cout << "Subgrahs size:\t" << this->Subgraphs.size() << std::endl;
  for ( unsigned int i=0; i<this->Subgraphs.size(); i++ )
    {
      //
      // DEB Check to see if the trachea node is present
      //
      double zMaxDEB = -DBL_MAX;

      std::cout << "Getting trachea node ID..." << std::endl;
      vtkIdType tracheaNodeId_DEB;
      std::cout << "num leaf nodes:\t" << this->Subgraphs[i].leafNodeIDs.size() << std::endl;
      for ( unsigned int j=0; j<this->Subgraphs[i].leafNodeIDs.size(); j++ )
  	{
	  vtkIdType nodeId_DEB = this->Subgraphs[i].leafNodeIDs[j];
	  unsigned int particleID_DEB = this->Subgraphs[i].nodeIDToParticleIDMap[nodeId_DEB];
	  if ( inputParticles->GetPoint(particleID_DEB)[2] > zMaxDEB)
	    {
	      zMaxDEB = inputParticles->GetPoint(particleID_DEB)[2];
	      tracheaNodeId_DEB = nodeId_DEB;
	    }
	}

      if ( this->Subgraphs[i].leafNodeIDs.size() > 0 )
	{
	  double maxScore = -DBL_MAX;     
	  vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
	      
	  std::cout << "---Getting trellis graph from subgraph..." << std::endl;
	  this->GetTrellisGraphFromSubgraph( &this->Subgraphs[i], tracheaNodeId_DEB, trellisGraph, inputParticles );
	  //this->ViewGraph( trellisGraph );
	  std::cout << "---DONE." << std::endl;

	  //
	  // If the labeling that the following function call determines is best so 
	  // far, record the labels in 'this->ParticleIDToGenerationLabel'
	  //
	  std::map< unsigned int, unsigned char > tmpParticleIDToGenerationLabelMap;
	      
	  std::cout << "---Computing generation labels from trellis graph..." << std::endl;
	  double score = this->ComputeGenerationLabelsFromTrellisGraph( trellisGraph, &tmpParticleIDToGenerationLabelMap );
	  std::cout << "---DONE." << std::endl;
	  //std::cout << "Score:\t" << score << std::endl;
	  if ( score > maxScore )
	    {
	      maxScore = score;
	      
	      this->UpdateAirwayGenerationAssignments( &tmpParticleIDToGenerationLabelMap );	    
	    }
  	}
      else
	{
	  double maxScore = -DBL_MAX;     
	  for ( unsigned int j=0; j<this->Subgraphs[i].leafNodeIDs.size(); j++ )
	    {
	      vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
	      
	      //std::cout << "---Getting trellis graph from subgraph..." << std::endl;
	      this->GetTrellisGraphFromSubgraph( &this->Subgraphs[i], this->Subgraphs[i].leafNodeIDs[j], trellisGraph, inputParticles );
	      //this->ViewGraph( trellisGraph );
	      //std::cout << "---DONE." << std::endl;

	      //
	      // If the labeling that the following function call determines is best so 
	      // far, record the labels in 'this->ParticleIDToGenerationLabel'
	      //
	      std::map< unsigned int, unsigned char > tmpParticleIDToGenerationLabelMap;
	      
	      std::cout << "---Computing generation labels from trellis graph..." << std::endl;
	      double score = this->ComputeGenerationLabelsFromTrellisGraph( trellisGraph, &tmpParticleIDToGenerationLabelMap );
	      std::cout << "---DONE." << std::endl;
	      //std::cout << "Score:\t" << score << std::endl;
	      if ( score > maxScore )
		{
		  maxScore = score;
	      
		  this->UpdateAirwayGenerationAssignments( &tmpParticleIDToGenerationLabelMap );
		}
	    }
	}
    }

  //
  // DEB: investigate the final assigments
  //
  {
    cip::ChestConventions conventions;

    std::map< unsigned int, unsigned char >::iterator mIt = this->ParticleIDToAirwayGenerationMap.begin();

    unsigned int total   = 0;
    unsigned int correct = 0;

    while ( mIt != this->ParticleIDToAirwayGenerationMap.end() )
      {	
  	unsigned char estState  = mIt->second;
  	unsigned char trueState = inputParticles->GetPointData()->GetArray( "ChestType" )->GetTuple( mIt->first )[0];

  	if ( estState == trueState )
  	  {
  	    correct++;
  	  }
  	total++;

  	mIt++;
      }

    std::cout << "Accuracy:\t" << static_cast< double >( correct )/static_cast< double >( total ) << std::endl;
  }

  //
  // At this point, all subgraphs have been considered and the best
  // generation labels have been determined. We can now fill out the 
  // output.
  //
  bool chestTypeArrayFound = false;

  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;

  for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inputParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inputParticles->GetPointData()->GetArray(i)->GetName() );

    arrayVec.push_back( array );

    if ( strcmp( inputParticles->GetPointData()->GetArray(i)->GetName(), "ChestType" ) == 0 )
      {
  	chestTypeArrayFound = true;
      }
    }

  if ( !chestTypeArrayFound )
    {
      vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
        array->SetNumberOfComponents( 1 );
  	array->SetName( "ChestType" );
      
      arrayVec.push_back( array );
    }

  vtkSmartPointer< vtkPoints > outputPoints  = vtkSmartPointer< vtkPoints >::New();  

  for ( unsigned int i=0; i<inputParticles->GetNumberOfPoints(); i++ )
    {
      outputPoints->InsertNextPoint( inputParticles->GetPoint(i) );

      for ( unsigned int k=0; k<arrayVec.size(); k++ )
  	{
  	  if ( strcmp( arrayVec[k]->GetName(), "ChestType" ) == 0 )
  	    {
  	      float state = static_cast< float >( this->ParticleIDToAirwayGenerationMap[i] );

  	      arrayVec[k]->InsertTuple( i, &state );
  	    }
  	  else
  	    {
  	      arrayVec[k]->InsertTuple( i, inputParticles->GetPointData()->GetArray(k)->GetTuple(i) );
  	    }
  	}     
    }
  
  outputParticles->SetPoints( outputPoints );
  for ( unsigned int j=0; j<arrayVec.size(); j++ )
    {
    outputParticles->GetPointData()->AddArray( arrayVec[j] );
    }

  return 1;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::AddAirwayGenerationLabeledAtlas( const vtkSmartPointer< vtkPolyData > atlas )
{
  this->AirwayGenerationLabeledAtlases.push_back( atlas );
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeAirwayGenerationAssignments( unsigned int numParticles )
{
  for ( unsigned int i=0; i<numParticles; i++ )
    {
      this->ParticleIDToAirwayGenerationMap[i] = cip::UNDEFINEDTYPE;
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetScaleStandardDeviation( unsigned char cipType, double std )
{
  this->ScaleStandardDeviationsMap[static_cast< float >( cipType )] = std;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetDistanceStandardDeviation( unsigned char cipType, double std )
{
  this->DistanceStandardDeviationsMap[static_cast< float >( cipType )] = std;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetAngleStandardDeviation( unsigned char cipType, double std )
{
  this->AngleStandardDeviationsMap[static_cast< float >( cipType )] = std;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetScaleMean( unsigned char cipType, double mean )
{
  this->ScaleMeansMap[static_cast< float >( cipType )] = mean;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetDistanceMean( unsigned char cipType, double mean )
{
  this->DistanceMeansMap[static_cast< float >( cipType )] = mean;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetAngleMean( unsigned char cipType, double mean )
{
  this->AngleMeansMap[static_cast< float >( cipType )] = mean;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeEmissionProbabilites( vtkSmartPointer< vtkPolyData > inputParticles )
{
  const double PI = 3.141592653589793238462;
  float  state;
  double mean, std, tmp; //Used for Kernel density estimation computation
  unsigned int correct = 0; //DEB

  //DEB
  std::vector< std::vector< unsigned int > > confusionMatrix;
  {
    for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      {
	std::vector< unsigned int > tmp;
	for ( unsigned int j=0; j<this->NumberOfStates; j++ )
	  {
	    tmp.push_back( 0 );
	  }
	confusionMatrix.push_back(tmp);
      }
  }

  //
  // For every particle in the unlabeled dataset, we'll compute the kernel
  // density estimated probability that it belongs to each airway generation 
  //
  for ( unsigned int p=0; p<this->NumberInputParticles; p++ )
    {
      std::map< unsigned char, double > probabilities;
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  probabilities[this->States[i]] = 1e-200;//DBL_MIN;
	}

      std::map< float, double >       probabilityAccumulatorMap;
      std::map< float, unsigned int > counterMap;
      //
      // Note that we loop from i=1 intentially, as i=0 corresponds
      // to noise (cip::UNDEFINEDTYPE)
      //
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  probabilityAccumulatorMap[static_cast< float >( this->States[i] )]  = 1e-200;//DBL_MIN;
	  counterMap[static_cast< float >( this->States[i] )]                 = 0;
	}

      float scale1 = inputParticles->GetPointData()->GetArray( "scale" )->GetTuple( p )[0];

      float point1[3];
        point1[0] = inputParticles->GetPoint( p )[0];
	point1[1] = inputParticles->GetPoint( p )[1];
	point1[2] = inputParticles->GetPoint( p )[2];

      double particle1Hevec2[3];
        particle1Hevec2[0] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[0];
	particle1Hevec2[1] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[1];
	particle1Hevec2[2] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[2];

      for ( unsigned int a=0; a<this->AirwayGenerationLabeledAtlases.size(); a++ )
	{
	  for ( unsigned int g=0; g<this->AirwayGenerationLabeledAtlases[a]->GetNumberOfPoints(); g++ )
	    {
	      state = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "ChestType" )->GetTuple( g )[0];
	      
	      float scale2 = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "scale" )->GetTuple( g )[0];
	      
	      float point2[3];
	        point2[0] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[0];
		point2[1] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[1];
		point2[2] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[2];

	      double connectingVec[3];
                connectingVec[0] = point1[0] - point2[0];
		connectingVec[1] = point1[1] - point2[1];
		connectingVec[2] = point1[2] - point2[2];

	      double distance = this->GetVectorMagnitude( connectingVec );

	      double particle2Hevec2[3];
  	        particle2Hevec2[0] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[0];
		particle2Hevec2[1] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[1];
		particle2Hevec2[2] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[2];

	      double angle1 =  this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
	      double angle2 =  this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

	      //
	      // Compute the kernel density estimation contribution. Note that it is the
	      // product of Guassians (for scale, position, and direction)
	      //
	      if ( distance < this->KernelDensityEstimationROIRadius )
		{
		  tmp  = 1;
		  std  = this->ScaleStandardDeviationsMap[state];
		  mean = this->ScaleMeansMap[state];
		  tmp  *= 1.0/(sqrt(2.0*PI)*std)*exp(-0.5*pow((scale1-scale2-mean)/std,2.0));

		  std  = this->DistanceStandardDeviationsMap[state];
		  mean = this->DistanceMeansMap[state];
		  tmp  *= 1.0/(sqrt(2.0*PI)*std)*exp(-0.5*pow((distance-mean)/std,2.0));
		  //tmp  *= 1.0/(sqrt(2.0*PI)*std)*exp(-0.5*pow(distance/std,2.0));

		  std  = this->AngleStandardDeviationsMap[state];
		  mean = this->AngleMeansMap[state];
		  tmp  *= 1.0/(sqrt(2.0*PI)*std)*exp(-0.5*pow((angle1-angle2-mean)/std,2.0));
		  probabilityAccumulatorMap[state] += tmp;

		  counterMap[state] += 1;
		}
	    }
	}
    
      //probabilities[this->States[0]] = 1e-200;//this->NoiseProbability;
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      	{      
      	  state = static_cast< float >( this->States[i] );
      	  if ( counterMap[state] > 0 )
      	    {
      	      probabilities[state] = probabilityAccumulatorMap[state]/static_cast< double >( counterMap[state] );	 
      	    }

	  // //DEB
	  // {
	  //   if ( state == inputParticles->GetPointData()->GetArray( "ChestType" )->GetTuple(p)[0] )
	  //     {
	  // 	probabilities[state] = 1.0;
	  //     }
	  //   else
	  //     {
	  // 	probabilities[state] = 1e-20;
	  //     }
	  // }
      	}
      
      //
      // Now normalize the probabilities so that they sum to one
      //
      double sum = 0.0;
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  sum += probabilities[this->States[i]];
	}
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  probabilities[this->States[i]] = probabilities[this->States[i]]/sum;
	}

      this->ParticleIDToEmissionProbabilitiesMap[p] = probabilities;

      //DEB
      {
      	double maxProb = 0.0;
      	unsigned int best;
	//std::cout << "-------------------------" << std::endl;
      	for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      	  {
      	    //  std::cout << probabilities[this->States[i]] << "\t";
      	    if ( probabilities[this->States[i]] > maxProb )
      	      {
      		maxProb = probabilities[this->States[i]];
      		best = this->States[i];
      	      }
      	  }

      	if ( best == inputParticles->GetPointData()->GetArray( "ChestType" )->GetTuple( p )[0] )
      	  {
      	    correct++;
      	  }

	unsigned int row = inputParticles->GetPointData()->GetArray( "ChestType" )->GetTuple( p )[0] - 38;
	unsigned int col = best -38;
	confusionMatrix[row][col] += 1;
      }	

    }

  //DEB
  {
    std::cout << "----------------- Confusion Matrix -----------------------" << std::endl;
    for ( unsigned int i=0; i<this->NumberOfStates; i++ )
      {
	for ( unsigned int j=0; j<this->NumberOfStates; j++ )
	  {
	    std::cout << confusionMatrix[i][j] << "\t";
	  }
	std::cout << std::endl;
      }
  }

  std::cout << "accuracy:\t" << static_cast< double >( correct )/static_cast< double >( this->NumberInputParticles ) << std::endl;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeSubGraphs( vtkSmartPointer< vtkMutableUndirectedGraph > spanningTree,
											 vtkSmartPointer< vtkPolyData > particles )
{
  vtkSmartPointer< vtkBoostConnectedComponents > connectedComponents = vtkSmartPointer< vtkBoostConnectedComponents >::New();
    connectedComponents->SetInputConnection( spanningTree->GetProducerPort() );
    connectedComponents->Update();
  
  vtkSmartPointer< vtkIntArray > components = 
    vtkIntArray::SafeDownCast( connectedComponents->GetOutput()->GetVertexData()->GetArray("component") );
      
  //
  // Get a unique list of the component numbers
  //
  std::list< unsigned int > componentNumberList;
  for( unsigned int i = 0; i < components->GetNumberOfTuples(); i++ )
    {
      componentNumberList.push_back( components->GetValue(i) ); 
    }
  componentNumberList.unique();
  componentNumberList.sort();
  componentNumberList.unique();

  //
  // Now fill out the subgraphs
  //
  std::list< unsigned int >::iterator it;
  for ( it = componentNumberList.begin(); it != componentNumberList.end(); it++ )
    {
      SUBGRAPH tempSubgraph;

      vtkSmartPointer< vtkIdTypeArray > ids = vtkSmartPointer<vtkIdTypeArray>::New();

      for( vtkIdType i = 0; i < components->GetNumberOfTuples(); i++ )
	{
	  if ( components->GetValue( i ) == *it )
	    {
	      ids->InsertNextValue( i );
	    }
	}
  
      vtkSmartPointer< vtkSelectionNode > nodes = vtkSmartPointer< vtkSelectionNode >::New();
        nodes->SetSelectionList( ids );
	nodes->SetContentType( vtkSelectionNode::INDICES );
	nodes->SetFieldType( vtkSelectionNode::VERTEX );

      vtkSmartPointer< vtkSelection > selection = vtkSmartPointer< vtkSelection >::New();
        selection->AddNode( nodes );
      
      vtkSmartPointer< vtkExtractSelectedGraph > extractSelectedGraph = vtkSmartPointer< vtkExtractSelectedGraph >::New();
        extractSelectedGraph->SetInput( 0, spanningTree );
	extractSelectedGraph->SetInput( 1, selection );
	extractSelectedGraph->Update();
     
      vtkSmartPointer< vtkMutableUndirectedGraph > subgraph = vtkSmartPointer< vtkMutableUndirectedGraph >::New();
	subgraph->ShallowCopy( extractSelectedGraph->GetOutput() );

      tempSubgraph.undirectedGraph = subgraph;

      this->Subgraphs.push_back( tempSubgraph );
    }

  //
  // For each subgraph, we want to associate the subgraphs node IDs to
  // the particle IDs. Do this by comparing the physical points associated
  // with the nodes in the subgraphs to the particle points. Also, determine
  // the leaf node IDs for each subgraph, as indicated by a vertex degree being
  // equal to 1.
  //
  for ( unsigned int i=0; i<this->Subgraphs.size(); i++ )
    {     
      for ( unsigned int j=0; j<this->Subgraphs[i].undirectedGraph->GetPoints()->GetNumberOfPoints(); j++ )
	{	
	  //
	  // Test if current subgraph node is leaf node
	  //
	  if ( this->Subgraphs[i].undirectedGraph->GetDegree(j) == 1 )
	    {
	      this->Subgraphs[i].leafNodeIDs.push_back(j);
	    }

	  //
	  // Find mapping between particle IDs and subgraph node IDs
	  //
	  for ( unsigned int k=0; k<this->NumberInputParticles; k++ )
	    {
	      if ( this->Subgraphs[i].undirectedGraph->GetPoint(j)[0] == particles->GetPoint(k)[0] &&
		   this->Subgraphs[i].undirectedGraph->GetPoint(j)[1] == particles->GetPoint(k)[1] &&
		   this->Subgraphs[i].undirectedGraph->GetPoint(j)[2] == particles->GetPoint(k)[2] )
		{
		  this->Subgraphs[i].nodeIDToParticleIDMap[j] = k;
		  this->Subgraphs[i].particleIDToNodeIDMap[k] = j;

		  break;
		}
	    }
	}
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > particles )
{
  //
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
  // 
  std::map< unsigned int, unsigned int > particleIDToNodeIDMap;
  std::map< unsigned int, unsigned int > nodeIDToParticleIDMap;

  vtkSmartPointer< vtkMutableUndirectedGraph > weightedGraph =  
    vtkSmartPointer< vtkMutableUndirectedGraph >::New();

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      vtkIdType nodeID = weightedGraph->AddVertex();

      particleIDToNodeIDMap[i]      = nodeID;
      nodeIDToParticleIDMap[nodeID] = i;
    }

  vtkSmartPointer< vtkDoubleArray > edgeWeights = vtkSmartPointer<vtkDoubleArray>::New();
    edgeWeights->SetNumberOfComponents( 1 );
    edgeWeights->SetName( "Weights" );

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      for ( unsigned int j=i+1; j<this->NumberInputParticles; j++ )
	{
	  double weight;
	  
	  if ( this->GetEdgeWeight( i, j, particles, &weight ) )
	    {
	      weightedGraph->AddEdge( particleIDToNodeIDMap[i], particleIDToNodeIDMap[j] );
	      edgeWeights->InsertNextValue( weight );
	    }
	}
    }

  weightedGraph->GetEdgeData()->AddArray( edgeWeights );
  weightedGraph->SetPoints( particles->GetPoints() );

  vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree > minimumSpanningTreeFilter = 
    vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree >::New();
    minimumSpanningTreeFilter->SetInput( weightedGraph );
    minimumSpanningTreeFilter->SetEdgeWeightArrayName( "Weights" );
    minimumSpanningTreeFilter->Update();

  vtkSmartPointer< vtkExtractSelectedGraph > extractSelection = vtkSmartPointer< vtkExtractSelectedGraph >::New();
    extractSelection->SetInput( 0, weightedGraph );
    extractSelection->SetInput( 1, minimumSpanningTreeFilter->GetOutput()) ;
    extractSelection->Update();

  this->MinimumSpanningTree = vtkMutableUndirectedGraph::SafeDownCast( extractSelection->GetOutput() );
}


bool vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetEdgeWeight( unsigned int particleID1, unsigned int particleID2, 
										   vtkSmartPointer< vtkPolyData > particles, double* weight )
{
  //
  // Determine the vector connecting the two particles
  //
  double point1[3];
    point1[0] = particles->GetPoint( particleID1 )[0];
    point1[1] = particles->GetPoint( particleID1 )[1];
    point1[2] = particles->GetPoint( particleID1 )[2];

  double point2[3];
    point2[0] = particles->GetPoint( particleID2 )[0];
    point2[1] = particles->GetPoint( particleID2 )[1];
    point2[2] = particles->GetPoint( particleID2 )[2];

  double connectingVec[3];
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = this->GetVectorMagnitude( connectingVec );

  if ( connectorMagnitude > this->ParticleDistanceThreshold )
    {
    return false;
    }

  double particle1Hevec2[3];
    particle1Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[0];
    particle1Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[1];
    particle1Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[2];

  double particle2Hevec2[3];
    particle2Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[0];
    particle2Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[1];
    particle2Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[2];

  double angle1 =  this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double angle2 =  this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

  if ( angle1 < angle2 )
    {
      *weight = connectorMagnitude*(1.0 + exp(-pow( (90.0 - angle1)/this->EdgeWeightAngleSigma, 2 )));
    }
  else
    {
      *weight = connectorMagnitude*(1.0 + exp(-pow( (90.0 - angle2)/this->EdgeWeightAngleSigma, 2 )));
    }

  return true;
}


double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetVectorMagnitude( double vector[3] )
{
  double magnitude = sqrt( pow( vector[0], 2 ) + pow( vector[1], 2 ) + pow( vector[2], 2 ) );

  return magnitude;
}


double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees )
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


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


//
// See Bishop's 'Pattern Recognition and Machine Learning', Chpt. 13.2 for a discussion
// of HMMs. Fig. 13.7 illustrates a trellis structure. It's through the trellis graph
// that we'll find the optimal path that determines the best assignment of airway
// generation states. We first need to create the trellis graph from a given
// (undirected) subgraph. The trellis is computed with respect to the specified leaf
// node ID. Each node in the trellis graph represents a hidden state (airway generation
// or noise). Edges represent transitions from one state to the next.
//
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetTrellisGraphFromSubgraph( SUBGRAPH* graph, vtkIdType leafNodeID, 
												 vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph,
												 vtkSmartPointer< vtkPolyData > particles )
{
  cip::ChestConventions conventionsDEB;

  const double PI = 3.141592653589793238462;

  //
  // For every node in the subgraph, we'll want 'N' nodes in the trellis
  // graph, where 'N' is the number of states. We need to keep track of 
  // which nodes have been "expanded" into trellis nodes
  //
  std::map< unsigned int, bool > graphNodeVisited;
  vtkSmartPointer< vtkVertexListIterator > vIt = vtkSmartPointer< vtkVertexListIterator >::New();
    graph->undirectedGraph->GetVertices( vIt );

  while ( vIt->HasNext() )
    {
      graphNodeVisited[vIt->Next()] = false;
    }

  // To each of these 'N' nodes, we want to associate the corresponding 
  // particle ID (same particle ID corresponding to the node in the subgraph) 
  // and a label indicating the latent state the trellis node represents. The 
  // particle IDs will be recorded in the trellis graphs 'particleIDArray', and 
  // the states will be recorded in the 'stateArray'. Furthermore, for every node 
  // in the trellis graph, we need to keep a record of the "best" parent node 
  // (keeping in mind that a node in the subgraph may have more than one
  // parent, and in that case we want to record the best trellis node
  // parent for each parent node in the subgraph), where "best" means
  // minimum cost incurred from the parent node and corresponding edge.
  // We'll keep a record of whether or not an edge emitting from a 
  // parent node is the "best" edge using a boolean valued array,
  // 'bestEdgeArray'. For a given parent node in the subgraph, there
  // should be one and only one 'true' valued edge from a parent node
  // in the trellis graph to a child node in the trellis graph.
  //
  vtkSmartPointer< vtkUnsignedCharArray > bestEdgeArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
    bestEdgeArray->SetNumberOfComponents( 1 );
    bestEdgeArray->SetName( "bestEdgeArray" );

  //
  // When we perform the forward search as part of the Viterbi algorithm,
  // we need to keep track of whether a node has been "passed through"
  // already. This needs to managed because several paths flowing 
  // through leaf nodes to the root node may pass through the same node.
  // The weight of that node must be set according to whether it has
  // already been passed through 
  //
  vtkSmartPointer< vtkUnsignedCharArray > visitedInForwardSearchArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
    visitedInForwardSearchArray->SetNumberOfComponents( 1 );
    visitedInForwardSearchArray->SetName( "visitedInForwardSearchArray" );

  vtkSmartPointer< vtkUnsignedIntArray > particleIDArray = vtkSmartPointer< vtkUnsignedIntArray >::New();
    particleIDArray->SetNumberOfComponents( 1 );
    particleIDArray->SetName( "particleIDArray" );

  vtkSmartPointer< vtkUnsignedCharArray > stateArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
    stateArray->SetNumberOfComponents( 1 );
    stateArray->SetName( "stateArray" );

  vtkSmartPointer< vtkFloatArray > edgeWeightArray = vtkSmartPointer< vtkFloatArray >::New();
    edgeWeightArray->SetNumberOfComponents( 1 );
    edgeWeightArray->SetName( "edgeWeightArray" );

  //
  // During the modified Viterbi algorithm, we perform a forward search in
  // order to find the most probable path through the trellis graph. Along the
  // way we compute accumulated weights at each node. The 'accumulatedWeightArray'
  // keeps track of these weights
  //
  vtkSmartPointer< vtkDoubleArray > accumulatedWeightArray = vtkSmartPointer< vtkDoubleArray >::New();
    accumulatedWeightArray->SetNumberOfComponents( 1 );
    accumulatedWeightArray->SetName( "accumulatedWeightArray" );

  trellisGraph->GetVertexData()->AddArray( particleIDArray );
  trellisGraph->GetVertexData()->AddArray( stateArray );
  trellisGraph->GetVertexData()->AddArray( accumulatedWeightArray );
  trellisGraph->GetVertexData()->AddArray( visitedInForwardSearchArray );
  trellisGraph->GetEdgeData()->AddArray( bestEdgeArray );
  trellisGraph->GetEdgeData()->AddArray( edgeWeightArray );

  //
  // This variable will be used below by the recursive routine 
  // 'AddStateNodesToTrellisGraph' to manage the connections 
  // (new edges) between parent nodes (held by this variable)
  // and the newly created state nodes.
  //
  std::vector< vtkIdType > trellisNodeIDGroup;

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {     
      vtkIdType id = trellisGraph->AddVertex();

      trellisNodeIDGroup.push_back( id );

      float tmpState      = static_cast< float >( this->States[i] );
      float tmpParticleID = static_cast< float >( (*graph).nodeIDToParticleIDMap[leafNodeID] );
      float tmpWeight     = 0.0;
      float tmpVisited    = 0.0;

      trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->InsertTuple( id, &tmpParticleID );
      trellisGraph->GetVertexData()->GetArray( "stateArray" )->InsertTuple( id, &tmpState );
      trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->InsertTuple( id, &tmpWeight );
      trellisGraph->GetVertexData()->GetArray( "visitedInForwardSearchArray" )->InsertTuple( id, &tmpVisited );

      graphNodeVisited[leafNodeID] = true;
    }

  vtkSmartPointer< vtkOutEdgeIterator > eIt = vtkSmartPointer< vtkOutEdgeIterator >::New();
  graph->undirectedGraph->GetOutEdges( leafNodeID, eIt );

  while( eIt->HasNext() )
    { 
      vtkOutEdgeType edge = eIt->Next();
      if ( !graphNodeVisited[edge.Target] )
	{
	  this->AddStateNodesToTrellisGraph( trellisGraph, trellisNodeIDGroup, edge.Target, graph, &graphNodeVisited );
	}
    }

  //
  // Now that we have the trellis graph constructed, we need to 
  // compute weights for each of the edges indicating the 
  // transition probability from one state to the next
  //
  vtkSmartPointer< vtkEdgeListIterator > trellisEdgeIt = vtkSmartPointer< vtkEdgeListIterator >::New();

  trellisGraph->GetEdges( trellisEdgeIt );

  double weight;
  while( trellisEdgeIt->HasNext() )
    {
      vtkEdgeType edge = trellisEdgeIt->Next();

      //
      // Get the particle ID of the source and target nodes
      //
      float sourceParticleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( edge.Source )[0];
      float targetParticleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( edge.Target )[0];
     
      //
      // Get the states of the source and target nodes
      //
      unsigned char sourceState = static_cast< unsigned char >( trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( edge.Source )[0] );
      unsigned char targetState = static_cast< unsigned char >( trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( edge.Target )[0] );

      //
      // Get the probability to assign to the edge
      //
      double probability;

      //
      // If the target node in 'graph' has more than two edges emanating from it,
      // then we are dealing with a branching, and we should use the branching
      // probabilities at this location.
      //
      vtkIdType targetGraphNodeID = graph->particleIDToNodeIDMap[targetParticleID];
      if ( graph->undirectedGraph->GetDegree( targetGraphNodeID ) > 2 )
	{
	  for ( unsigned int i=0; i<this->BranchingTransitionProbabilities.size(); i++ )
	    {
	      if ( this->BranchingTransitionProbabilities[i].sourceState == sourceState &&
		   this->BranchingTransitionProbabilities[i].targetState == targetState )
		{
		  probability = this->BranchingTransitionProbabilities[i].probability;
		  break;
		}
	    }
	}
      else
	{	  
	  probability = this->GetTransitionProbability( sourceParticleID, targetParticleID, sourceState, targetState, particles );
	}

      trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->InsertTuple( edge.Id, &probability );
    }

  //
  // The 'while' loop just executed sets preliminary probabilities for each of
  // the edges in the trellis graph. These must now be normalized. The probability
  // of a given node (state) in the trellis diagram transitioning to *something* is
  // 1.0. Thus, we must loop over all edges emanating from each node and normalize
  // their values so they add up to 1.0.
  //
  vtkSmartPointer< vtkVertexListIterator > trellisVertexIt = vtkSmartPointer< vtkVertexListIterator >::New();
  trellisGraph->GetVertices( trellisVertexIt );

  while ( trellisVertexIt->HasNext() )
    {
      vtkIdType stateNodeID = trellisVertexIt->Next();

      //
      // Loop over all the edges emanating from this node
      //
      vtkSmartPointer< vtkOutEdgeIterator > outEdgeIt = vtkSmartPointer< vtkOutEdgeIterator >::New();
      trellisGraph->GetOutEdges( stateNodeID, outEdgeIt );

      double accum = 0.0;
      while ( outEdgeIt->HasNext() )
	{
	  vtkOutEdgeType edge = outEdgeIt->Next();

	  accum += trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0];
	}

      //
      // Now loop over the edges again, and normalize the weights so that
      // they add up to one 
      //
      trellisGraph->GetOutEdges( stateNodeID, outEdgeIt );
      while ( outEdgeIt->HasNext() )
	{
	  vtkOutEdgeType edge = outEdgeIt->Next();

	  double newWeight = trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0]/accum;
	  trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->SetTuple( edge.Id, &newWeight );
	}
    }
}


double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::ComputeGenerationLabelsFromTrellisGraph( vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph, 
													       std::map< unsigned int, unsigned char >* particleIDToGenerationLabel )
{
  //
  // Loop through all nodes and collect node IDs for those that have
  // no parents. This is the set of non-root leaf nodes, and it's where we
  // begin the forward search
  //
  vtkSmartPointer< vtkVertexListIterator > vIt = vtkSmartPointer< vtkVertexListIterator >::New();
  trellisGraph->GetVertices( vIt );

  std::vector< vtkIdType > nodeVec;
  while ( vIt->HasNext() )
    {
      vtkIdType nodeID = vIt->Next();
      if ( this->IsNonRootLeafNode( nodeID, trellisGraph ) )
	{
	  nodeVec.push_back( nodeID );

	  //
	  // The "accumulatedWeight" at these root nodes will just be the log of the
	  // emission probabilities
	  //
	  unsigned char state               = trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
	  float         particleID          = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
	  float         visited             = 1.0;
	  double        emissionProbability = this->ParticleIDToEmissionProbabilitiesMap[static_cast< unsigned int >( particleID )][state];

	  double weight;
	  if ( emissionProbability == 0.0 )
	    {
	      weight = -1e200;//-DBL_MAX;
	    }
	  else
	    {
	      weight = log( emissionProbability );
	    }

	  trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->SetTuple( nodeID, &weight );
	  trellisGraph->GetVertexData()->GetArray( "visitedInForwardSearchArray" )->SetTuple( nodeID, &visited );
	}
    }

  //
  // Now get all nodes pointed to by the root nodes
  //
  std::list< vtkIdType > nodeList;
  for ( unsigned int i=0; i<nodeVec.size(); i++ )
    {
      vtkSmartPointer< vtkOutEdgeIterator > it = vtkSmartPointer< vtkOutEdgeIterator >::New();
      trellisGraph->GetOutEdges( nodeVec[i], it );
      
      while( it->HasNext() )
  	{
  	  vtkIdType tmpID = it->Next().Target;
  	  nodeList.push_back( tmpID );
  	}
    }
  nodeList.unique();
  nodeList.sort();
  nodeList.unique();

  //
  // The 'bestFinalStateNodeID' will be set to the best root state node
  // in the trellis graph. It will be the terminal node at the end of the
  // path representing the most probable sequence of states, and it will
  // be used to initiate the backtracking procedure below.
  //
  vtkIdType bestFinalStateNodeID;

  while ( nodeList.size() > 0 )
    {
      //
      // Now for each node in 'nodeList', we want to update the trellis graph
      // with accumulated weights along various paths through trellis. This
      // is accomplished by considering one node at a time: for a given node
      // we consider all incoming edges (grouping edges according to their 
      // association with distinct particles). For a given edge group, we
      // identify the edge that incurs the greatest probability amongst all
      // other edges in the group. Once the best edge has been identified
      // for each edge group, we mark those edges as being "optimal", and
      // we update the weight at the node under consideration.
      //
      std::list< vtkIdType >::iterator listIt = nodeList.begin();
      while ( listIt != nodeList.end() )
	{
	  this->UpdateTrellisGraphWithViterbiStep( *listIt, trellisGraph );
	  
	  listIt++;
	}

      //
      // Now we want to collect all the nodes being pointed to by
      // the nodes in 'nodeList' for the next go-around
      //
      std::list< vtkIdType > tmpNodeList;
      listIt = nodeList.begin();
      while ( listIt != nodeList.end() )
	{
	  vtkSmartPointer< vtkOutEdgeIterator > it = vtkSmartPointer< vtkOutEdgeIterator >::New();
	  trellisGraph->GetOutEdges( *listIt, it );
	  
	  while( it->HasNext() )
	    {
	      vtkIdType tmpID = it->Next().Target;
	      tmpNodeList.push_back( tmpID );
	    }

	  listIt++;
	}
      tmpNodeList.unique();
      tmpNodeList.sort();
      tmpNodeList.unique();

      //
      // If 'tmpNodeList' is empty, that means we have reached the root node of 
      // the trellis. At this point we have everything we need to back-track through
      // the trellis to identify the most probable states. First we identify the 
      // state node that has the greatest accumulated weight. This node is at the
      // end of the best path through the trellis, and its weight is the highest
      // accumulated weight of all paths through the trellis. We will use this
      // node ID to initiate the back-tracking below.
      //
      if ( tmpNodeList.size() == 0 )
	{
	  double maxPathWeight = -DBL_MAX;

	  listIt = nodeList.begin();
	  while ( listIt != nodeList.end() )
	    {
	      double pathWeight = trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->GetTuple( *listIt )[0];

	      if ( pathWeight > maxPathWeight )
		{
		  maxPathWeight        = pathWeight;
		  bestFinalStateNodeID = *listIt;
		}

	      listIt++;
	    }
	}

      //
      // Copy the contents of 'tmpNodeList' into the 'nodeList' container
      //
      nodeList.clear();
      listIt = tmpNodeList.begin();
      while ( listIt != tmpNodeList.end() )
	{
	  nodeList.push_back( *listIt );
	  
	  listIt++;
	}
    }

  //
  // Finally, backtrack through the trellis graph to identify the 
  // the best states for each of the particles. Also compute the score 
  // corresponding to this path. Note that the score we compute is different
  // than the simple accumulated weight along the path. 
  //
  double score = 0;

  this->BackTrack( bestFinalStateNodeID, trellisGraph, particleIDToGenerationLabel, &score );

  return score;
}


//
// This is a recursive routine to back-track through the trellis graph
// along the most probable path (paths, in the case of bifurcation occurencs),
// to identify the most probable state sequence
//
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::BackTrack( vtkIdType nodeID, vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph, 
									       std::map< unsigned int, unsigned char >* particleIDToGenerationLabel, double* score )
{
  unsigned int  particleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
  unsigned char state      = trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
  double        emissionProbability = this->ParticleIDToEmissionProbabilitiesMap[particleID][state];

  if ( emissionProbability == 0.0 )
    {
      *score += -1e200;
    }
  else
    {
      *score += log( emissionProbability );
    }

  (*particleIDToGenerationLabel)[particleID] = state;

  vtkSmartPointer< vtkInEdgeIterator > it = vtkSmartPointer< vtkInEdgeIterator >::New();
  trellisGraph->GetInEdges( nodeID, it );

  while ( it->HasNext() )
    {
      vtkInEdgeType edge = it->Next();
      if ( trellisGraph->GetEdgeData()->GetArray( "bestEdgeArray" )->GetTuple( edge.Id )[0] == 1.0 )
	{
	  double edgeProbability = trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0];
	  if ( edgeProbability == 0.0 )
	    {
	      *score += -1e200;
	    }
	  else
	    {
	      *score += log( edgeProbability );
	    }
	  //std::cout << "---Score after edge prob update:\t" << *score << std::endl;
	  this->BackTrack( edge.Source, trellisGraph, particleIDToGenerationLabel, score );
	}
    }
}


//
// The Viterbi algorithm is used to find the most probable sequence of states.
// We march through trellis graph updating the accumulated weights at each state
// node and keeping track of the incoming edges to a given state that have the
// greatest weight. This function does this update at a single state node location.
//
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::UpdateTrellisGraphWithViterbiStep( vtkIdType nodeID, vtkSmartPointer< vtkMutableDirectedGraph > graph )
{
  //
  // Get all edges incoming. We want to consider the weight we will
  // incur by "choosing" a given edge: that weight is the sum of the
  // ln transition probability associated with the edge and the weight
  // already accumulated at the source state node. Note in general we
  // may have more than one particle "flowing into" the current node.
  // Thus, we want to keep track of the optimal weight and the best
  // edge for each particle. The "best edge" is the one, for a given
  // particle, that contributes the highest weight. Recording these
  // edges will allow us to backtrack through the trellis graph once
  // we reach the end of the forward search in order to identify the
  // most probable sequence of states.
  // 
  vtkSmartPointer< vtkInEdgeIterator > it = vtkSmartPointer< vtkInEdgeIterator >::New();
  graph->GetInEdges( nodeID, it );

  //
  // These maps allow us to distinguish from possibly
  // multiple particles "flowing into" a given node
  //
  std::map< float, double >     weightMap;
  std::map< float, vtkIdType >  bestEdgeMap;

  while ( it->HasNext() )
    {
      vtkInEdgeType edge = it->Next();

      float best = 0.0;
      graph->GetEdgeData()->GetArray( "bestEdgeArray" )->SetTuple( edge.Id, &best );

      float particleID = graph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( edge.Source )[0];

      double edgeProbability   = graph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0];
      double accumulatedWeight = graph->GetVertexData()->GetArray( "accumulatedWeightArray" )->GetTuple( edge.Source )[0];

      double totWeight;
      if ( edgeProbability == 0.0 )
	{
	  //totWeight = -DBL_MAX + accumulatedWeight;
	  totWeight = -1e200 + accumulatedWeight;
	}
      else
	{
	  totWeight = log( edgeProbability ) + accumulatedWeight;
	}

      bool found = false;
      std::map< float, double >::iterator mIt = weightMap.begin();
      while ( mIt != weightMap.end() )
	{
	  if ( mIt->first == particleID )
	    {
	      found = true;

	      if ( totWeight > weightMap[particleID] )
		{
		  weightMap[particleID]   = totWeight;
		  bestEdgeMap[particleID] = edge.Id;
		}
	    }

	  mIt++;
	}
      
      if ( !found )
	{
	  weightMap[particleID]   = totWeight;
	  bestEdgeMap[particleID] = edge.Id;
	}
    }

  //
  // At this point we have identified the best weights and edges. We 
  // now want to update the current node's weight, which is the sum 
  // of the ln of the emission probability and the weights we found above
  //
  unsigned char state                  = graph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
  float         particleID             = graph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
  float         visitedInForwardSearch = graph->GetVertexData()->GetArray( "visitedInForwardSearchArray" )->GetTuple( nodeID )[0];
  double        emissionProbability    = this->ParticleIDToEmissionProbabilitiesMap[static_cast< unsigned int >( particleID )][state];

  double weightAccumulator = 0.0;
  if ( visitedInForwardSearch == 1.0 )
    {
      weightAccumulator += graph->GetVertexData()->GetArray( "accumulatedWeightArray" )->GetTuple( nodeID )[0];
    }
  else
    {
      float visited = 1.0;
      graph->GetVertexData()->GetArray( "visitedInForwardSearchArray" )->SetTuple( nodeID, &visited );

      if ( emissionProbability == 0.0 )
	{
	  weightAccumulator = -1e200;//-DBL_MAX;
	}
      else
	{
	  weightAccumulator = log( emissionProbability );
	}
    }

  std::map< float, double >::iterator wIt = weightMap.begin();
  while ( wIt != weightMap.end() )
    {
      weightAccumulator += wIt->second;
      wIt++;
    }
  
  graph->GetVertexData()->GetArray( "accumulatedWeightArray" )->SetTuple( nodeID, &weightAccumulator );
  
  //
  // Finally, mark the "best" edges as being the best. This will be
  // used in the back-tracking stage later.
  //
  std::map< float, vtkIdType >::iterator bIt = bestEdgeMap.begin();
  while ( bIt != bestEdgeMap.end() )
    {
      float best = 1.0;

      graph->GetEdgeData()->GetArray( "bestEdgeArray" )->SetTuple( bIt->second, &best );
      
      bIt++;
    }  
}


//
// Root nodes should have no incoming edges
//
bool vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::IsNonRootLeafNode( vtkIdType nodeID, vtkSmartPointer< vtkMutableDirectedGraph > graph )
{
  vtkSmartPointer< vtkInEdgeIterator > it = vtkSmartPointer< vtkInEdgeIterator >::New();
  graph->GetInEdges( nodeID, it );
 
  bool nonRootLeafNode = true;
  while( it->HasNext() )
    { 
      nonRootLeafNode = false;
      break;
    }

  return nonRootLeafNode;
}


unsigned int vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetStateIndex( unsigned char state )
{
  unsigned int index;

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
      if ( this->States[i] == state )
	{
	  index = i;
	  break;
	}
    }

  return index;
}


double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetTransitionProbability( unsigned int sourceParticleID, unsigned int targetParticleID,
												unsigned char sourceState, unsigned int targetState,
												vtkSmartPointer< vtkPolyData > particles ) 
{
  cip::ChestConventions conventionsDEB;

  //DEB
  bool printDEB = false;
  {
    float fromState = particles->GetPointData()->GetArray( "ChestType" )->GetTuple( sourceParticleID )[0];
    float toState   = particles->GetPointData()->GetArray( "ChestType" )->GetTuple( targetParticleID )[0];
    
    if ( fromState == float(cip::MAINBRONCHUS) && toState == float(cip::TRACHEA) &&
	 int(sourceState) == int(cip::MAINBRONCHUS) && int(targetState) == int(cip::TRACHEA) )
      {
	printDEB = true;
      }
  }

  const double PI = 3.141592653589793238462;

  double probability;

  bool found = false;
  for ( unsigned int i=0; i<this->NormalTransitionProbabilityParameters.size(); i++ )
    {
      if ( this->NormalTransitionProbabilityParameters[i].sourceState == sourceState &&
	   this->NormalTransitionProbabilityParameters[i].targetState == targetState )
	{
	  found = true;

	  double sourceDirection[3];
	  sourceDirection[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[0];
	  sourceDirection[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[1];
	  sourceDirection[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[2];
	  
	  double targetDirection[3];
	  targetDirection[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[0];
	  targetDirection[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[1];
	  targetDirection[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[2];
	  
	  double angle = this->GetAngleBetweenVectors( sourceDirection, targetDirection, true );
	  
	  double sourceScale =  particles->GetPointData()->GetArray( "scale" )->GetTuple( sourceParticleID )[0];
	  double targetScale =  particles->GetPointData()->GetArray( "scale" )->GetTuple( targetParticleID )[0];
	  
	  double scaleDiff = targetScale-sourceScale;

	  if ( printDEB )
	    {
	      std::cout << "angle:\t" << angle << std::endl;
	      std::cout << "sourceScale:\t" << sourceScale << std::endl;
	      std::cout << "targetScale:\t" << targetScale << std::endl;
	    }
	  
	  double meanSc  = this->NormalTransitionProbabilityParameters[i].scaleDifferenceMean;
	  if ( printDEB )
	    {
	      std::cout << "meanSc:\t" << meanSc << std::endl;
	    }
	  double stdSc   = sqrt( this->NormalTransitionProbabilityParameters[i].scaleDifferenceVariance );
	  if ( printDEB )
	    {
	      std::cout << "stdSc:\t" << meanSc << std::endl;
	    }
	  double meanAng = this->NormalTransitionProbabilityParameters[i].angleMean;
	  if ( printDEB )
	    {
	      std::cout << "meanAng:\t" << meanAng << std::endl;
	    }
	  double stdAng  = sqrt( this->NormalTransitionProbabilityParameters[i].angleVariance );
	  if ( printDEB )
	    {
	      std::cout << "stdAng:\t" << stdAng << std::endl;
	    }
	  double N1 = 1.0/(sqrt(2.0*PI)*stdSc)*exp(-0.5*pow((scaleDiff-meanSc)/stdSc,2.0));
	  double N2 = 1.0/(sqrt(2.0*PI)*stdAng)*exp(-0.5*pow((angle-meanAng)/stdAng,2.0));
	  if ( printDEB )
	    {
	      std::cout << "N1:\t" << N1 << std::endl;
	      std::cout << "N2:\t" << N1 << std::endl;
	    }

	  probability = N1*N2;

	  break;
	}
    }

  if ( !found )
    {
      if ( printDEB )
	{
	  std::cout << "----------------------------" << std::endl;
	  std::cout << "Not found??" << std::endl;
	  std::cout << conventionsDEB.GetChestTypeName(sourceState) << std::endl;
	  std::cout << conventionsDEB.GetChestTypeName(targetState) << std::endl;
	  std::cout << "----------------------------" << std::endl;
	}
      for ( unsigned int i=0; i<this->TransitionProbabilities.size(); i++ )
	{
	  if ( this->TransitionProbabilities[i].sourceState == sourceState &&
	       this->TransitionProbabilities[i].targetState == targetState )
	    {
	      probability = this->TransitionProbabilities[i].probability;

	      break;
	    }
	}
    }

  return probability;
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::AddStateNodesToTrellisGraph( vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph, 
												 std::vector< vtkIdType > trellisNodeIDGroup, vtkIdType subgraphID,
												 SUBGRAPH* graph, std::map< unsigned int, bool >* graphNodeVisited )
{
  std::vector< vtkIdType > tmpTrellisNodeIDGroup;

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {     
      vtkIdType id = trellisGraph->AddVertex();
      
      tmpTrellisNodeIDGroup.push_back( id );

      float tmpState      = static_cast< float >( this->States[i] );
      float tmpParticleID = static_cast< float >( (*graph).nodeIDToParticleIDMap[subgraphID] );
      float tmpWeight     = 0.0;
      float tmpVisited    = 0.0;

      trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->InsertTuple( id, &tmpParticleID );
      trellisGraph->GetVertexData()->GetArray( "stateArray" )->InsertTuple( id, &tmpState );
      trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->InsertTuple( id, &tmpWeight );      
      trellisGraph->GetVertexData()->GetArray( "visitedInForwardSearchArray" )->InsertTuple( id, &tmpVisited );      

      (*graphNodeVisited)[subgraphID] = true;
    }

  //
  // Now create directed edges from all newly created nodes to the 
  // nodes in the group passed to this function
  //
  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
      vtkIdType pID = tmpTrellisNodeIDGroup[i];

      for ( unsigned int j=0; j<this->NumberOfStates; j++ )
	{
	  vtkIdType cID = trellisNodeIDGroup[j];

	  vtkEdgeType edge = trellisGraph->AddEdge( pID, cID );

	  float best = 0;
	  trellisGraph->GetEdgeData()->GetArray( "bestEdgeArray" )->InsertTuple( edge.Id, &best );
	}
    }

  //
  // Now loop over all the edges emanating from this node. If
  // a node in the subgraph hasn't been visited, it is a child
  // node of the current node and needs to be "expanded" in
  // the trellis graph
  //
  vtkSmartPointer< vtkOutEdgeIterator > eIt = vtkSmartPointer< vtkOutEdgeIterator >::New();
  graph->undirectedGraph->GetOutEdges( subgraphID, eIt );

  while( eIt->HasNext() )
    { 
      vtkOutEdgeType edge = eIt->Next();
      if ( !(*graphNodeVisited)[edge.Target] )
	{
	  this->AddStateNodesToTrellisGraph( trellisGraph, tmpTrellisNodeIDGroup, edge.Target, graph, graphNodeVisited );
	}
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::UpdateAirwayGenerationAssignments( std::map< unsigned int, unsigned char >* tmpParticleIDToGenerationLabelMap )
{
  std::map< unsigned int, unsigned char >::iterator mIt = (*tmpParticleIDToGenerationLabelMap).begin();
  while ( mIt != (*tmpParticleIDToGenerationLabelMap).end() )
    {
      this->ParticleIDToAirwayGenerationMap[mIt->first] = mIt->second;

      ++mIt;
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::ViewGraph( vtkSmartPointer<vtkMutableUndirectedGraph> graph )
{ 
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(600, 300);

  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  vtkSmartPointer< vtkGraphLayoutView > graphLayoutView = vtkSmartPointer<vtkGraphLayoutView>::New();
    graphLayoutView->AddRepresentationFromInput( graph );
    graphLayoutView->SetRenderWindow (renderWindow );
    graphLayoutView->SetInteractor( renderWindowInteractor );
    graphLayoutView->SetEdgeLabelVisibility( true );
    graphLayoutView->SetEdgeLabelArrayName( "Weights" );
    graphLayoutView->ResetCamera();
    graphLayoutView->Render();

  renderWindowInteractor->Start();
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::ViewGraph( vtkSmartPointer<vtkMutableDirectedGraph> graph )
{ 
  vtkSmartPointer< vtkSimple2DLayoutStrategy > strategy = vtkSmartPointer< vtkSimple2DLayoutStrategy >::New();
 
  vtkSmartPointer< vtkGraphLayout > layout =  vtkSmartPointer< vtkGraphLayout >::New();
    layout->SetInput( graph );
    layout->SetLayoutStrategy( strategy );
  
  vtkSmartPointer< vtkGraphToPolyData > graphToPoly = vtkSmartPointer< vtkGraphToPolyData >::New();
    graphToPoly->SetInputConnection( layout->GetOutputPort() );
    graphToPoly->EdgeGlyphOutputOn();
    graphToPoly->SetEdgeGlyphPosition(0.98);
 
  vtkSmartPointer< vtkGlyphSource2D > arrowSource = vtkSmartPointer< vtkGlyphSource2D >::New();
    arrowSource->SetGlyphTypeToEdgeArrow();
    arrowSource->SetScale(0.1);
    arrowSource->Update();
 
  vtkSmartPointer< vtkGlyph3D > arrowGlyph = vtkSmartPointer< vtkGlyph3D >::New();
    arrowGlyph->SetInputConnection( 0, graphToPoly->GetOutputPort(1) );
    arrowGlyph->SetInputConnection( 1, arrowSource->GetOutputPort()) ;
 
  vtkSmartPointer< vtkPolyDataMapper > arrowMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    arrowMapper->SetInputConnection(arrowGlyph->GetOutputPort());

  vtkSmartPointer< vtkActor > arrowActor =  vtkSmartPointer< vtkActor >::New();
    arrowActor->SetMapper(arrowMapper);

  vtkSmartPointer< vtkGraphLayoutView > graphLayoutView = vtkSmartPointer< vtkGraphLayoutView >::New();
    graphLayoutView->SetLayoutStrategyToPassThrough();
    graphLayoutView->SetEdgeLayoutStrategyToPassThrough(); 
    graphLayoutView->AddRepresentationFromInputConnection( layout->GetOutputPort() );
    graphLayoutView->GetRenderer()->AddActor( arrowActor ); 
    graphLayoutView->ResetCamera();
    graphLayoutView->Render();
    graphLayoutView->GetInteractor()->Start();
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::ViewPolyData( vtkSmartPointer< vtkMutableUndirectedGraph > graph )
{
  // vtkIndent indent;
  // polyData->PrintSelf( std::cout, indent );

  vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
    graphToPolyData->SetInput( graph );
    graphToPolyData->Update();

  // vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
  // writer->SetInput( graphToPolyData->GetOutput() );
  // writer->SetFileName( "/Users/jross/tmp/test.vtk" );
  // writer->Write();

  vtkSmartPointer< vtkRenderer > renderer = vtkSmartPointer< vtkRenderer >::New();
    renderer->SetBackground( 1, 1, 1 ); 

  vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
  mapper->SetInputConnection( graphToPolyData->GetOutputPort() );
 
  vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
    actor->SetMapper( mapper );
    actor->GetProperty()->SetColor( 0, 0, 0 );

  renderer->AddActor( actor );

  vtkSmartPointer< vtkRenderWindow > renderWindow = vtkSmartPointer< vtkRenderWindow >::New();
    renderWindow->AddRenderer( renderer );

  vtkSmartPointer< vtkInteractorStyleTrackballCamera > trackball = vtkSmartPointer< vtkInteractorStyleTrackballCamera >::New();

  vtkSmartPointer< vtkRenderWindowInteractor > renderWindowInteractor = vtkSmartPointer< vtkRenderWindowInteractor >::New();
    renderWindowInteractor->SetRenderWindow( renderWindow );
    renderWindowInteractor->SetInteractorStyle( trackball );

  //
  // Set up the nodes to be rendered
  //
  vtkSmartPointer< vtkSphereSource > sphereSource = vtkSmartPointer< vtkSphereSource >::New();
    sphereSource->SetRadius( 0.2 );
    sphereSource->SetCenter( 0, 0, 0 );

//   vtkSmartPointer< vtkPoints > leafPoints = vtkSmartPointer< vtkPoints >::New();
//   for ( unsigned int i=0; i<this->SubGraphLeafParticleIDs.size(); i++ )
//     {
//     unsigned int leafParticleID  = this->SubGraphLeafParticleIDs[i];

//     double leafPoint[3];
//     leafPoint[0] = this->InternalInputPolyData->GetPoint( leafParticleID )[0];
//     leafPoint[1] = this->InternalInputPolyData->GetPoint( leafParticleID )[1];
//     leafPoint[2] = this->InternalInputPolyData->GetPoint( leafParticleID )[2];

//     leafPoints->InsertNextPoint( leafPoint[0], leafPoint[1], leafPoint[2] );
//     }

  vtkSmartPointer< vtkPolyData > nodesPoly = vtkSmartPointer< vtkPolyData >::New();
    nodesPoly->SetPoints( graph->GetPoints() );

  vtkSmartPointer< vtkGlyph3D > nodesGlyph = vtkSmartPointer< vtkGlyph3D >::New();
    nodesGlyph->SetInput( nodesPoly );
    nodesGlyph->SetSource( sphereSource->GetOutput() );
    nodesGlyph->Update();

  vtkSmartPointer< vtkPolyDataMapper > nodesMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    nodesMapper->SetInput( nodesGlyph->GetOutput() );

  vtkSmartPointer<vtkActor> nodesActor = vtkSmartPointer<vtkActor>::New();
    nodesActor->SetMapper( nodesMapper );
    nodesActor->GetProperty()->SetColor( 1, 0, 0 );

  renderer->AddActor( nodesActor );

  renderWindow->Render();
  renderWindowInteractor->Start();
}
