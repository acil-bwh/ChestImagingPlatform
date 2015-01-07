/**
 *
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 */

#include "vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.h"

#include "cipChestConventions.h"
#include "vtkMath.h"
#include "vtkIdTypeArray.h"
#include "vtkSelectionNode.h"
#include "vtkIndent.h"
#include "vtkExtractSelectedGraph.h"
#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "vtkVertexListIterator.h"
#include "vtkMutableUndirectedGraph.h"
#include "vtkBoostConnectedComponents.h"
#include "vtkUnsignedIntArray.h"
#include "vtkOutEdgeIterator.h"
#include "vtkEdgeListIterator.h"
#include "vtkInEdgeIterator.h"
#include "cipHelper.h"
#include <cfloat>
#include <math.h>
#include <list>


vtkStandardNewMacro( vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter );
 
vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter()
{
  this->EdgeWeightAngleSigma                    = 1.0;
  this->NoiseProbability                        = DBL_MIN; //TODO: How best to handle noise?
  this->KernelDensityEstimationROIRadius        = DBL_MAX;
  this->NumberOfStates                          = 11; //Noise is not currently handled

  this->HMTMMode = true;

  this->ParticleRootNodeID = -1; // Negative indicates no root node has been specified.

  // The values for computing the emission probabilities. These values were learned
  // from training data. Note that technically the distribution for angle has finite
  // support (from 0 to 90 degrees), whereas the exponential distribution chose to
  // model it has infinite support. However, the chosen value of 0.06 causes the 
  // distribution to taper down to nearly 0 and 90 degrees, so it's a reasonable
  // approximation.
  this->EmissionDistanceLambda = 0.32679;
  this->EmissionScaleMu        = 0.0;
  this->EmissionScaleSigma     = 0.787;
  this->EmissionAngleLambda    = 0.06;

  // The following values for the transition likelihood terms were learned from
  // training data
  this->SameTransitionScaleMu     = 0.0;
  this->SameTransitionScaleSigma  = 0.1514;
  this->SameTransitionAngleLambda = 0.13;

  this->DiffTransitionScaleMu1        = 0.0436;
  this->DiffTransitionScaleSigma1     = 0.582;
  this->DiffTransitionScaleWeight1    = 0.68;
  this->DiffTransitionScaleMu2        = 0.8568;
  this->DiffTransitionScaleSigma2     = 0.804;
  this->DiffTransitionScaleWeight2    = 0.32;
  this->DiffTransitionAngleSlope1     = 0.00071;
  this->DiffTransitionAngleSlope2     = -0.0002029;
  this->DiffTransitionAngleIntercept1 = 0.004;
  this->DiffTransitionAngleIntercept2 = 0.022261;

  this->States.push_back( static_cast< unsigned char >( cip::TRACHEA ) );
  this->States.push_back( static_cast< unsigned char >( cip::MAINBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::SUPERIORDIVISIONBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::LINGULARBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::MIDDLELOBEBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::INTERMEDIATEBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::LOWERLOBEBRONCHUS ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ) );
  this->States.push_back( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ) );

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {
      for ( unsigned int j=0; j<this->NumberOfStates; j++ )
	{
	  TRANSITIONPROBABILITY transProb;
	  transProb.sourceState = this->States[i];
	  transProb.targetState = this->States[j];

	  if ( (this->States[i] >= this->States[j]) && this->States[i] != (unsigned char)( cip::UNDEFINEDTYPE ) && i-j <= 2 )
	    {
	      transProb.probability = 1e-200;
	    }
	  else if ( this->States[i] == (unsigned char)( cip::UNDEFINEDTYPE ) ||
		    this->States[j] == (unsigned char)( cip::UNDEFINEDTYPE ) )
	    {
	      transProb.probability = 1e-200;
	    }
	  else
	    {
	      transProb.probability = 0.0;
	    }
	  
	  this->TransitionProbabilities.push_back( transProb );

	  // Use transition probability priors learned from training data, which, when multipiplied
	  // by the probability of scale difference and angle given the transition is proportional 
	  // to the probability of the transition given scale difference and angle
	  transProb.probability = 0.0;
	  this->TransitionProbabilityPriors.push_back( transProb );
	}
    }

  // Now we set the transition probability priors explicitly as learned from training data.
  for ( unsigned int i=0; i<this->TransitionProbabilityPriors.size(); i++ )
    {
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION5) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::AIRWAYGENERATION5) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.942707789;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION5) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::AIRWAYGENERATION4) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.057292211;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION4) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::AIRWAYGENERATION4) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.945667643;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION4) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::AIRWAYGENERATION3) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.054332357;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::AIRWAYGENERATION3) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.937587413;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.007867133;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.006468531;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::LOWERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.034440559;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.006293706;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::AIRWAYGENERATION3) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::LINGULARBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.007342657;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.943661972;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::UPPERLOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.056338028;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.03133515;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.96866485;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.957317073;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::MIDDLELOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::LOWERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.042682927;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::LOWERLOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::LOWERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.978074866;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::LOWERLOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::INTERMEDIATEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.013368984;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::LOWERLOBEBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.00855615;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::MAINBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::MAINBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.978555305;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::MAINBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::TRACHEA) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.021444695;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.92733564;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.07266436;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::LINGULARBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::LINGULARBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.951871658;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::LINGULARBRONCHUS) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::UPPERLOBEBRONCHUS) )
	{
	  this->TransitionProbabilityPriors[i].probability = 0.048128342;
	}
      if ( this->TransitionProbabilityPriors[i].sourceState == (unsigned char)(cip::TRACHEA) &&
	   this->TransitionProbabilityPriors[i].targetState == (unsigned char)(cip::TRACHEA) )
	{
	  this->TransitionProbabilityPriors[i].probability = 1.0;
	}
    }        

  this->SetNumberOfInputPorts( 1 );
  this->SetNumberOfOutputPorts( 1 );
}

void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetParticleRootNodeID( unsigned int nodeID )
{
  this->ParticleRootNodeID = int(nodeID);
}

vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::~vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter()
{  
}

void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetModeToHMTM()
{
  this->HMTMMode = true;
}
 
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::SetModeToKDE()
{
  this->HMTMMode = false;
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

  // Initialize the airway generation assignments. Each particle
  // will be assigned cip::UNDEFINEDTYPE to initialize. Calling
  // this function will fill 'this->ParticleIDToAirwayGenerationMap'
  // for subsequent use.
  this->InitializeAirwayGenerationAssignments( this->NumberInputParticles );
  
  // Compute the emission probabilities. We only need to do this once. 
  // The emission probabilites are independent of the graph structure
  // of the particles. Calling this function will fill out
  // 'this->ParticleIDToEmissionProbabilitiesMap' for subsequent use.
  std::cout << "---Initializing emission probs..." << std::endl;
  this->InitializeEmissionProbabilites( inputParticles );
  std::cout << "---DONE." << std::endl;  

  if ( this->HMTMMode )
    {
      // Compute the minimum spanning tree that will be used to establish
      // directionality through the particles. Calling this function will 
      // fill out 'this->MinimumSpanningTree' for subsequent use
      std::cout << "---Initializing minimum spanning tree..." << std::endl;
      this->InitializeMinimumSpanningTree( inputParticles );
      std::cout << "---DONE." << std::endl;
      //cip::ViewGraphAsPolyData( this->MinimumSpanningTree );

      // Now that we have initialized the minimum spanning tree, we can
      // construct the subgraphs (in general it make be the case that
      // several subgraphs / subtrees exist. We will want to comptue the 
      // HMM labeler for all of them). Calling this function will fill
      // out 'this->Subgraphs' for subsequent use.
      std::cout << "---Initializing subgraphs..." << std::endl;
      this->InitializeSubGraphs( this->MinimumSpanningTree, inputParticles );
      std::cout << "---DONE." << std::endl;

      // Now for each subgraph, consider each leaf node in turn, assume
      // it is the root node, and perform the labeling. Whichever leaf node
      // acts as the most probable root node (based on the greatest likelihood)
      // will be considered the true root node for that subtree, and the 
      // corresponding generation labels will be used.
      for ( unsigned int i=0; i<this->Subgraphs.size(); i++ )
	{
	  // If the root node is known and specified, use it. Otherwise, find the leaf node most likely
	  // to the root node
	  bool rootNodeFound = false;
	  if ( this->ParticleRootNodeID >= 0 )
	    {
	      // Do a quick check to see if the specified root node is in fact a valid
	      // leaf node for the subgraph being considered
	      for ( unsigned int j=0; j<this->Subgraphs[i].leafNodeIDs.size(); j++ )
		{
		  vtkIdType nodeId = this->Subgraphs[i].leafNodeIDs[j];
		  unsigned int particleID = this->Subgraphs[i].nodeIDToParticleIDMap[nodeId];

		  if ( particleID == this->ParticleRootNodeID )
		    {
		      rootNodeFound = true;
		      break;
		    }
		}

	      if ( rootNodeFound )
		{
		  vtkIdType rootNodeID = this->Subgraphs[i].particleIDToNodeIDMap[this->ParticleRootNodeID];		  
		  vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
		  
		  std::cout << "---Getting trellis graph from subgraph..." << std::endl;
		  this->GetTrellisGraphFromSubgraph( &this->Subgraphs[i], rootNodeID, trellisGraph, inputParticles );
		  //cip::ViewGraph( trellisGraph );
		  std::cout << "---DONE." << std::endl;

		  std::map< unsigned int, unsigned char > tmpParticleIDToGenerationLabelMap;
	      
		  std::cout << "---Computing generation labels from trellis graph..." << std::endl;
		  this->ComputeGenerationLabelsFromTrellisGraph( trellisGraph, &tmpParticleIDToGenerationLabelMap );
		  std::cout << "---DONE." << std::endl;

		  std::cout << "---Updating airway generation assignments..." << std::endl;
		  this->UpdateAirwayGenerationAssignments( &tmpParticleIDToGenerationLabelMap );	    	
		  std::cout << "---DONE." << std::endl;
		}
	      else
		{
		  std::cout << "WARNING: Root node specified, but not found" << std::endl;
		}
	    }
	  else if ( !rootNodeFound )
	    {
	      double maxScore = -DBL_MAX;     
	      for ( unsigned int j=0; j<this->Subgraphs[i].leafNodeIDs.size(); j++ )
	        {
	          vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
	      
	          std::cout << "---Getting trellis graph from subgraph..." << std::endl;
	          this->GetTrellisGraphFromSubgraph( &this->Subgraphs[i], this->Subgraphs[i].leafNodeIDs[j], trellisGraph, inputParticles );
	          std::cout << "---DONE." << std::endl;
	      
	          // If the labeling that the following function call determines is best so 
	          // far, record the labels in 'this->ParticleIDToGenerationLabel'
	          std::map< unsigned int, unsigned char > tmpParticleIDToGenerationLabelMap;
	      
	          std::cout << "---Computing generation labels from trellis graph..." << std::endl;
	          double score = this->ComputeGenerationLabelsFromTrellisGraph( trellisGraph, &tmpParticleIDToGenerationLabelMap );
	          std::cout << "---DONE." << std::endl;

	          if ( score > maxScore )
		    {
		      maxScore = score;		      
		      this->UpdateAirwayGenerationAssignments( &tmpParticleIDToGenerationLabelMap );
		    }
	        }
	    }
	}      
    }

  // At this point, 'ParticleIDToAirwayGenerationMap' should be up to date, either by applying KDE based
  // classification or by applying the complete HMTM algorithm. We can now fill out the output.
  std::vector< vtkSmartPointer< vtkFloatArray > > arrayVec;
  
  for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
    {
    vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
      array->SetNumberOfComponents( inputParticles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
      array->SetName( inputParticles->GetPointData()->GetArray(i)->GetName() );

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

void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeEmissionProbabilites( vtkSmartPointer< vtkPolyData > inputParticles )
{
  const double PI = 3.141592653589793238462;
  float  state;
  double mean, std, tmp; //Used for Kernel density estimation computation

  // For every particle in the unlabeled dataset, we'll compute the kernel
  // density estimated probability that it belongs to each airway generation 
  for ( unsigned int p=0; p<this->NumberInputParticles; p++ )
    {
      std::map< unsigned char, double > probabilities;
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  probabilities[this->States[i]] = 1e-100;//DBL_MIN;
	}

      std::map< float, double >       probabilityAccumulatorMap;
      std::map< float, unsigned int > counterMap;

      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  probabilityAccumulatorMap[static_cast< float >( this->States[i] )]  = 1e-100;//DBL_MIN;
	  counterMap[static_cast< float >( this->States[i] )]                 = 0;
	}

      float scale1 = inputParticles->GetPointData()->GetArray( "scale" )->GetTuple( p )[0];

      float point1[3];
        point1[0] = inputParticles->GetPoint( p )[0];
	point1[1] = inputParticles->GetPoint( p )[1];
	point1[2] = inputParticles->GetPoint( p )[2];

      cip::VectorType particle1Hevec2(3);
        particle1Hevec2[0] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[0];
	particle1Hevec2[1] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[1];
	particle1Hevec2[2] = inputParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( p )[2];

      for ( unsigned int a=0; a<this->AirwayGenerationLabeledAtlases.size(); a++ )
	{
	  for ( unsigned int g=0; g<this->AirwayGenerationLabeledAtlases[a]->GetNumberOfPoints(); g++ )
	    {
	      state = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "ChestType" )->GetTuple( g )[0];
	      
	      float scale2 = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "scale" )->GetTuple( g )[0];
	      
	      cip::PointType point2(3);
	        point2[0] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[0];
		point2[1] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[1];
		point2[2] = this->AirwayGenerationLabeledAtlases[a]->GetPoint( g )[2];

	      cip::VectorType connectingVec(3);
                connectingVec[0] = point1[0] - point2[0];
		connectingVec[1] = point1[1] - point2[1];
		connectingVec[2] = point1[2] - point2[2];

	      double distance = cip::GetVectorMagnitude( connectingVec );

	      cip::VectorType particle2Hevec2(3);
  	        particle2Hevec2[0] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[0];
		particle2Hevec2[1] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[1];
		particle2Hevec2[2] = this->AirwayGenerationLabeledAtlases[a]->GetPointData()->GetArray( "hevec2" )->GetTuple( g )[2];

	      double angle =  cip::GetAngleBetweenVectors( particle1Hevec2, particle2Hevec2, true );

	      // Compute the kernel density estimation contribution. Note that it is the
	      // product of Guassians (for scale, position, and direction)
	      if ( distance < this->KernelDensityEstimationROIRadius )
		{		  
		  tmp  = 1.0;

		  // Compute the scale contribution
		  tmp  *= 1.0/(sqrt(2.0*PI)*this->EmissionScaleSigma)*exp(-0.5*pow((scale1-scale2-this->EmissionScaleMu)/this->EmissionScaleSigma, 2.0));

		  // Compute the distance contribution
		  tmp *= this->EmissionDistanceLambda*exp( -this->EmissionDistanceLambda*angle );

		  // Compute the angle contribution
		  tmp *= this->EmissionAngleLambda*exp( -this->EmissionAngleLambda*angle );

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
      	}
      
      // Now normalize the probabilities so that they sum to one
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

      // Now update 'ParticleIDToAirwayGenerationMap' in case the user wants to assign 
      // generation labels based only on KDE-based classification      
      double maxProb = 0.0;
      unsigned int best;
      for ( unsigned int i=0; i<this->NumberOfStates; i++ )
	{
	  if ( probabilities[this->States[i]] > maxProb )
	    {
	      maxProb = probabilities[this->States[i]];
	      best = this->States[i];
	    }
	}
      this->ParticleIDToAirwayGenerationMap[p] = (unsigned char)(best);
    }
}


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::InitializeSubGraphs( vtkSmartPointer< vtkMutableUndirectedGraph > spanningTree,
											 vtkSmartPointer< vtkPolyData > particles )
{
  vtkSmartPointer< vtkBoostConnectedComponents > connectedComponents = vtkSmartPointer< vtkBoostConnectedComponents >::New();
    connectedComponents->SetInputData( spanningTree );
    connectedComponents->Update();
  
  vtkSmartPointer< vtkIntArray > components = 
    vtkIntArray::SafeDownCast( connectedComponents->GetOutput()->GetVertexData()->GetArray("component") );
      
  // Get a unique list of the component numbers
  std::list< unsigned int > componentNumberList;
  for( unsigned int i = 0; i < components->GetNumberOfTuples(); i++ )
    {
      componentNumberList.push_back( components->GetValue(i) ); 
    }
  componentNumberList.unique();
  componentNumberList.sort();
  componentNumberList.unique();

  // Now fill out the subgraphs
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
        extractSelectedGraph->SetInputData( 0, spanningTree );
	extractSelectedGraph->SetInputData( 1, selection );
	extractSelectedGraph->Update();
     
      vtkSmartPointer< vtkMutableUndirectedGraph > subgraph = vtkSmartPointer< vtkMutableUndirectedGraph >::New();
	subgraph->ShallowCopy( extractSelectedGraph->GetOutput() );

      tempSubgraph.undirectedGraph = subgraph;

      this->Subgraphs.push_back( tempSubgraph );
    }

  // For each subgraph, we want to associate the subgraphs node IDs to
  // the particle IDs. Do this by comparing the physical points associated
  // with the nodes in the subgraphs to the particle points. Also, determine
  // the leaf node IDs for each subgraph, as indicated by a vertex degree being
  // equal to 1.
  for ( unsigned int i=0; i<this->Subgraphs.size(); i++ )
    {     
      for ( unsigned int j=0; j<this->Subgraphs[i].undirectedGraph->GetPoints()->GetNumberOfPoints(); j++ )
	{	
	  // Test if current subgraph node is leaf node
	  if ( this->Subgraphs[i].undirectedGraph->GetDegree(j) == 1 )
	    {
	      this->Subgraphs[i].leafNodeIDs.push_back(j);
	    }

	  // Find mapping between particle IDs and subgraph node IDs
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
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
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
    minimumSpanningTreeFilter->SetInputData( weightedGraph );
    minimumSpanningTreeFilter->SetEdgeWeightArrayName( "Weights" );
    minimumSpanningTreeFilter->Update();

  vtkSmartPointer< vtkExtractSelectedGraph > extractSelection = vtkSmartPointer< vtkExtractSelectedGraph >::New();
    extractSelection->SetInputData( 0, weightedGraph );
    extractSelection->SetInputData( 1, minimumSpanningTreeFilter->GetOutput()) ;
    extractSelection->Update();

  this->MinimumSpanningTree = vtkMutableUndirectedGraph::SafeDownCast( extractSelection->GetOutput() );
}


bool vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetEdgeWeight( unsigned int particleID1, unsigned int particleID2, 
										   vtkSmartPointer< vtkPolyData > particles, double* weight )
{
  // Determine the vector connecting the two particles
  double point1[3];
    point1[0] = particles->GetPoint( particleID1 )[0];
    point1[1] = particles->GetPoint( particleID1 )[1];
    point1[2] = particles->GetPoint( particleID1 )[2];

  double point2[3];
    point2[0] = particles->GetPoint( particleID2 )[0];
    point2[1] = particles->GetPoint( particleID2 )[1];
    point2[2] = particles->GetPoint( particleID2 )[2];

  cip::VectorType connectingVec(3);
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = cip::GetVectorMagnitude( connectingVec );

  if ( connectorMagnitude > this->ParticleDistanceThreshold )
    {
    return false;
    }

  cip::VectorType particle1Hevec2(3);
    particle1Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[0];
    particle1Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[1];
    particle1Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[2];

  cip::VectorType particle2Hevec2(3);
    particle2Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[0];
    particle2Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[1];
    particle2Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[2];

  double angle1 =  cip::GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double angle2 =  cip::GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

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


void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


// See Bishop's 'Pattern Recognition and Machine Learning', Chpt. 13.2 for a discussion
// of HMMs. Fig. 13.7 illustrates a trellis structure. It's through the trellis graph
// that we'll find the optimal path that determines the best assignment of airway
// generation states. We first need to create the trellis graph from a given
// (undirected) subgraph. The trellis is computed with respect to the specified leaf
// node ID. Each node in the trellis graph represents a hidden state (airway generation
// or noise). Edges represent transitions from one state to the next.
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetTrellisGraphFromSubgraph( SUBGRAPH* graph, vtkIdType leafNodeID, 
												 vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph,
												 vtkSmartPointer< vtkPolyData > particles )
{
  const double PI = 3.141592653589793238462;

  // For every node in the subgraph, we'll want 'N' nodes in the trellis
  // graph, where 'N' is the number of states. We need to keep track of 
  // which nodes have been "expanded" into trellis nodes
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
  vtkSmartPointer< vtkUnsignedCharArray > bestEdgeArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
    bestEdgeArray->SetNumberOfComponents( 1 );
    bestEdgeArray->SetName( "bestEdgeArray" );

  vtkSmartPointer< vtkUnsignedIntArray > particleIDArray = vtkSmartPointer< vtkUnsignedIntArray >::New();
    particleIDArray->SetNumberOfComponents( 1 );
    particleIDArray->SetName( "particleIDArray" );

  vtkSmartPointer< vtkUnsignedCharArray > stateArray = vtkSmartPointer< vtkUnsignedCharArray >::New();
    stateArray->SetNumberOfComponents( 1 );
    stateArray->SetName( "stateArray" );

  vtkSmartPointer< vtkFloatArray > edgeWeightArray = vtkSmartPointer< vtkFloatArray >::New();
    edgeWeightArray->SetNumberOfComponents( 1 );
    edgeWeightArray->SetName( "edgeWeightArray" );

  // The following array will keep a record of whether or not a given trellis node has an
  // accumulated weight amount that is up to date (that is, all possible incoming paths
  // have been considered)
  vtkSmartPointer< vtkFloatArray > upToDateForForwardSearchArray = vtkSmartPointer< vtkFloatArray >::New();
    upToDateForForwardSearchArray->SetNumberOfComponents( 1 );
    upToDateForForwardSearchArray->SetName( "upToDateForForwardSearchArray" );

  // During the modified Viterbi algorithm, we perform a forward search in
  // order to find the most probable path through the trellis graph. Along the
  // way we compute accumulated weights at each node. The 'accumulatedWeightArray'
  // keeps track of these weights
  vtkSmartPointer< vtkDoubleArray > accumulatedWeightArray = vtkSmartPointer< vtkDoubleArray >::New();
    accumulatedWeightArray->SetNumberOfComponents( 1 );
    accumulatedWeightArray->SetName( "accumulatedWeightArray" );

  trellisGraph->GetVertexData()->AddArray( particleIDArray );
  trellisGraph->GetVertexData()->AddArray( stateArray );
  trellisGraph->GetVertexData()->AddArray( accumulatedWeightArray );
  trellisGraph->GetVertexData()->AddArray( upToDateForForwardSearchArray );
  trellisGraph->GetEdgeData()->AddArray( bestEdgeArray );
  trellisGraph->GetEdgeData()->AddArray( edgeWeightArray );

  // This variable will be used below by the recursive routine 
  // 'AddStateNodesToTrellisGraph' to manage the connections 
  // (new edges) between parent nodes (held by this variable)
  // and the newly created state nodes.
  std::vector< vtkIdType > trellisNodeIDGroup;

  for ( unsigned int i=0; i<this->NumberOfStates; i++ )
    {     
      vtkIdType id = trellisGraph->AddVertex();

      trellisNodeIDGroup.push_back( id );

      float tmpState      = static_cast< float >( this->States[i] );
      float tmpParticleID = static_cast< float >( (*graph).nodeIDToParticleIDMap[leafNodeID] );
      float tmpWeight     = 0.0;
      float tmpVisited    = 0.0;
      float tmpUpToDate   = 0.0;

      trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->InsertTuple( id, &tmpParticleID );
      trellisGraph->GetVertexData()->GetArray( "stateArray" )->InsertTuple( id, &tmpState );
      trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->InsertTuple( id, &tmpWeight );
      trellisGraph->GetVertexData()->GetArray( "upToDateForForwardSearchArray" )->InsertTuple( id, &tmpUpToDate );

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

  // Now that we have the trellis graph constructed, we need to 
  // compute weights for each of the edges indicating the 
  // transition probability from one state to the next
  vtkSmartPointer< vtkEdgeListIterator > trellisEdgeIt = vtkSmartPointer< vtkEdgeListIterator >::New();

  trellisGraph->GetEdges( trellisEdgeIt );

  double weight;
  while( trellisEdgeIt->HasNext() )
    {
      vtkEdgeType edge = trellisEdgeIt->Next();

      // Get the particle ID of the source and target nodes
      float sourceParticleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( edge.Source )[0];
      float targetParticleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( edge.Target )[0];
     
      // Get the states of the source and target nodes
      unsigned char sourceState = static_cast< unsigned char >( trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( edge.Source )[0] );
      unsigned char targetState = static_cast< unsigned char >( trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( edge.Target )[0] );

      // Get the probability to assign to the edge     
      vtkIdType targetGraphNodeID = graph->particleIDToNodeIDMap[targetParticleID];
      double probability = this->GetTransitionProbability( sourceParticleID, targetParticleID, sourceState, targetState, particles );
      trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->InsertTuple( edge.Id, &probability );
    }

  // The 'while' loop just executed sets preliminary probabilities for each of
  // the edges in the trellis graph. These must now be normalized. The probability
  // of a given node (state) in the trellis diagram transitioning to *something* is
  // 1.0. Thus, we must loop over all edges emanating from each node and normalize
  // their values so they add up to 1.0.
  vtkSmartPointer< vtkVertexListIterator > trellisVertexIt = vtkSmartPointer< vtkVertexListIterator >::New();
  trellisGraph->GetVertices( trellisVertexIt );

  while ( trellisVertexIt->HasNext() )
    {
      vtkIdType stateNodeID = trellisVertexIt->Next();

      // Loop over all the edges emanating from this node
      vtkSmartPointer< vtkOutEdgeIterator > outEdgeIt = vtkSmartPointer< vtkOutEdgeIterator >::New();
      trellisGraph->GetOutEdges( stateNodeID, outEdgeIt );

      double accum = 0.0;
      while ( outEdgeIt->HasNext() )
	{	 
	  vtkOutEdgeType edge = outEdgeIt->Next();

	  accum += trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0];
	}

      // Now loop over the edges again, and normalize the weights so that
      // they add up to one 
      trellisGraph->GetOutEdges( stateNodeID, outEdgeIt );
      while ( outEdgeIt->HasNext() )
	{
	  vtkOutEdgeType edge = outEdgeIt->Next();

	  double newWeight;
	  if ( accum == 0.0 )
	    {
	      newWeight = 1.0/double(this->NumberOfStates);
	    }
	  else
	    {
	      newWeight = trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->GetTuple( edge.Id )[0]/accum;
	    }

	  trellisGraph->GetEdgeData()->GetArray( "edgeWeightArray" )->SetTuple( edge.Id, &newWeight );
	}
    }
}

double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::ComputeGenerationLabelsFromTrellisGraph( vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph, 
													       std::map< unsigned int, unsigned char >* particleIDToGenerationLabel )
{
  // Loop through all nodes and collect node IDs for those that have
  // no parents. This is the set of non-root leaf nodes, and it's where we
  // begin the forward search
  vtkSmartPointer< vtkVertexListIterator > vIt = vtkSmartPointer< vtkVertexListIterator >::New();
  trellisGraph->GetVertices( vIt );

  std::vector< vtkIdType > nodeVec;
  while ( vIt->HasNext() )
    {
      vtkIdType nodeID = vIt->Next();
      if ( this->IsNonRootLeafNode( nodeID, trellisGraph ) )
  	{
  	  nodeVec.push_back( nodeID );

  	  // The "accumulatedWeight" at these root nodes will just be the log of the
  	  // emission probabilities
  	  unsigned char state               = trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
  	  float         particleID          = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
  	  float         visited             = 1.0;
  	  float         upToDate            = 1.0;
  	  double        emissionProbability = this->ParticleIDToEmissionProbabilitiesMap[static_cast< unsigned int >( particleID )][state];

  	  double weight;
  	  if ( emissionProbability < 1e-200 )
  	    {
  	      weight = -1e100;//-DBL_MAX;
  	    }
  	  else
  	    {
  	      weight = log( emissionProbability );
  	    }

  	  trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->SetTuple( nodeID, &weight );
  	  trellisGraph->GetVertexData()->GetArray( "upToDateForForwardSearchArray" )->SetTuple( nodeID, &upToDate );
  	}
    }

  // Now get all nodes pointed to by the leaf nodes
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

  // The 'bestFinalStateNodeID' will be set to the best root state node
  // in the trellis graph. It will be the terminal node at the end of the
  // path representing the most probable sequence of states, and it will
  // be used to initiate the backtracking procedure below.
  vtkIdType bestFinalStateNodeID;

  while ( nodeList.size() > 0 )
    {
      // Now for each node in 'nodeList', we want to update the trellis graph
      // with accumulated weights along various paths through trellis. This
      // is accomplished by considering one node at a time: for a given node
      // we consider all incoming edges (grouping edges according to their 
      // association with distinct particles). For a given edge group, we
      // identify the edge that incurs the greatest probability amongst all
      // other edges in the group. Once the best edge has been identified
      // for each edge group, we mark those edges as being "optimal", and
      // we update the weight at the node under consideration.      
      std::list< vtkIdType >::iterator listIt = nodeList.begin();

      // Keep a list of trellis node IDs that need to be considered for the 
      // next go around
      std::list< vtkIdType > tmpNodeList;

      while ( listIt != nodeList.end() )
  	{
  	  // If the call to 'UpdateTrellisGraphWithViterbiStep' returns true, that means that all
  	  // incoming edges to the query node have associated source nodes that are all up to date.
  	  // This means that after the call to this function, the query node itself is made up
  	  // to date. In that case, we can collect all the nodes being pointed to by this query
  	  // node. If, on the other hand, the funtion returns false, that means that the query node
  	  // has not been made up to date because it is waiting on other nodes on which it 
  	  // depends to be made up to date. In that case, we want to add the query node back into
  	  // the basket of nodes that needs to be considered again for the next go around.
  	  if ( this->UpdateTrellisGraphWithViterbiStep( *listIt, trellisGraph ) )
  	    {
  	      vtkSmartPointer< vtkOutEdgeIterator > it = vtkSmartPointer< vtkOutEdgeIterator >::New();
  	      trellisGraph->GetOutEdges( *listIt, it );

  	      while( it->HasNext() )
  		{
  		  vtkIdType tmpID = it->Next().Target;
  		  tmpNodeList.push_back( tmpID );
  		}
  	    }
	  
  	  listIt++;
  	}
      tmpNodeList.unique();
      tmpNodeList.sort();
      tmpNodeList.unique();

      // If 'tmpNodeList' is empty, that means we have reached the root node of 
      // the trellis. At this point we have everything we need to back-track through
      // the trellis to identify the most probable states. First we identify the 
      // state node that has the greatest accumulated weight. This node is at the
      // end of the best path through the trellis, and its weight is the highest
      // accumulated weight of all paths through the trellis. We will use this
      // node ID to initiate the back-tracking below.
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

      // Copy the contents of 'tmpNodeList' into the 'nodeList' container
      nodeList.clear();
      listIt = tmpNodeList.begin();
      while ( listIt != tmpNodeList.end() )
  	{
  	  nodeList.push_back( *listIt );
	  
  	  listIt++;
  	}
    }

  // Finally, backtrack through the trellis graph to identify the 
  // the best states for each of the particles. Also compute the score 
  // corresponding to this path. Note that the score we compute is different
  // than the simple accumulated weight along the path. 
  double score = 0;

  this->BackTrack( bestFinalStateNodeID, trellisGraph, particleIDToGenerationLabel, &score );

  return score;
}

// This is a recursive routine to back-track through the trellis graph
// along the most probable path (paths, in the case of bifurcation occurencs),
// to identify the most probable state sequence
void vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::BackTrack( vtkIdType nodeID, vtkSmartPointer< vtkMutableDirectedGraph > trellisGraph, 
									       std::map< unsigned int, unsigned char >* particleIDToGenerationLabel, double* score )
{
  unsigned int  particleID = trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
  unsigned char state      = trellisGraph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
  double        emissionProbability = this->ParticleIDToEmissionProbabilitiesMap[particleID][state];

  if ( emissionProbability < 1e-200 )
    {
      *score += -1e100;
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
	  if ( edgeProbability < 1e-200 )
	    {
	      *score += -1e100;
	    }
	  else
	    {
	      *score += log( edgeProbability );
	    }
	  this->BackTrack( edge.Source, trellisGraph, particleIDToGenerationLabel, score );
	}
    }
}

// The Viterbi algorithm is used to find the most probable sequence of states.
// We march through trellis graph updating the accumulated weights at each state
// node and keep track of the incoming edges to a given state that have the
// greatest weight. This function does this update at a single state node location.
bool vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::UpdateTrellisGraphWithViterbiStep( vtkIdType nodeID, vtkSmartPointer< vtkMutableDirectedGraph > graph )
{
  // First check that the source node for every incoming edge into the query node
  // is up to date. If any one of them is not, do nothing and return false (we will
  // come back to this node later when all the nodes on which it depends are up to
  // date).
  vtkSmartPointer< vtkInEdgeIterator > itCheck = vtkSmartPointer< vtkInEdgeIterator >::New();
  graph->GetInEdges( nodeID, itCheck );

  while ( itCheck->HasNext() )
    {
      vtkInEdgeType edge = itCheck->Next();
      double upToDate = graph->GetVertexData()->GetArray( "upToDateForForwardSearchArray" )->GetTuple( edge.Source )[0];

      if ( upToDate == 0.0 )
	{
	  return false;
	}
    }

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
  vtkSmartPointer< vtkInEdgeIterator > it = vtkSmartPointer< vtkInEdgeIterator >::New();
  graph->GetInEdges( nodeID, it );

  // These maps allow us to distinguish from possibly
  // multiple particles "flowing into" a given node
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
      if ( edgeProbability < 1e-200 )
	{
	  //totWeight = -DBL_MAX + accumulatedWeight;
	  totWeight = -1e100 + accumulatedWeight;
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

  // At this point we have identified the best weights and edges. We 
  // now want to update the current node's weight, which is the sum 
  // of the ln of the emission probability and the weights we found above
  unsigned char state                  = graph->GetVertexData()->GetArray( "stateArray" )->GetTuple( nodeID )[0];
  float         particleID             = graph->GetVertexData()->GetArray( "particleIDArray" )->GetTuple( nodeID )[0];
  double        emissionProbability    = this->ParticleIDToEmissionProbabilitiesMap[static_cast< unsigned int >( particleID )][state];

  double weightAccumulator = 0.0;
  if ( emissionProbability < 1e-200 )
    {
      weightAccumulator = -1e100;
    }
  else
    {
      weightAccumulator = log( emissionProbability );
    }

  std::map< float, double >::iterator wIt = weightMap.begin();
  while ( wIt != weightMap.end() )
    {
      weightAccumulator += wIt->second;
      wIt++;
    }
  
  graph->GetVertexData()->GetArray( "accumulatedWeightArray" )->SetTuple( nodeID, &weightAccumulator );
  
  // Mark the "best" edges as being the best. This will be used in the back-tracking stage later.
  std::map< float, vtkIdType >::iterator bIt = bestEdgeMap.begin();
  while ( bIt != bestEdgeMap.end() )
    {
      float best = 1.0;

      graph->GetEdgeData()->GetArray( "bestEdgeArray" )->SetTuple( bIt->second, &best );
      
      bIt++;
    }  

  // Lastly, mark this node as being up to date
  double upToDate = 1.0;
  graph->GetVertexData()->GetArray( "upToDateForForwardSearchArray" )->SetTuple( nodeID, &upToDate );

  return true;
}

// Leaf nodes should have no incoming edges
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

double vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter::GetTransitionProbability( unsigned int sourceParticleID, unsigned int targetParticleID,
												unsigned char sourceState, unsigned int targetState,
												vtkSmartPointer< vtkPolyData > particles ) 
{
  const double PI = 3.141592653589793238462;

  double prior = 0.0;
  for ( unsigned int i=0; i<this->TransitionProbabilityPriors.size(); i++ )
    {
      if ( this->TransitionProbabilityPriors[i].sourceState == sourceState &&
	   this->TransitionProbabilityPriors[i].targetState == targetState )
	{
	  prior = this->TransitionProbabilityPriors[i].probability;
	  break;
	}
    }

  // Now that we have the prior, compute the likelihood. The posterior (transition given scale and
  // angle) will then be proportional to the product of the two.
  cip::VectorType sourceDirection(3);
    sourceDirection[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[0];
    sourceDirection[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[1];
    sourceDirection[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( sourceParticleID )[2];
	  
  cip::VectorType targetDirection(3);
    targetDirection[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[0];
    targetDirection[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[1];
    targetDirection[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( targetParticleID )[2];
	  
  double angle = cip::GetAngleBetweenVectors( sourceDirection, targetDirection, true );
	  
  double scaleDiff = particles->GetPointData()->GetArray( "scale" )->GetTuple( sourceParticleID )[0] -
    particles->GetPointData()->GetArray( "scale" )->GetTuple( targetParticleID )[0];

  // We use different likelihood functions depending on whether it is a "same state" transition or
  // a "different state" transition (learned from data).
  double likelihood = 1.0;
  if ( sourceState == targetState )
    {
      // Compute the scale contribution
      likelihood *= 1.0/(sqrt(2.0*PI)*this->SameTransitionScaleSigma)*exp(-0.5*pow((scaleDiff - this->SameTransitionScaleMu)/this->SameTransitionScaleSigma, 2.0));

      // Compute the angle contribution
      likelihood *= this->SameTransitionAngleLambda*exp( -this->SameTransitionAngleLambda*angle );
    }
  else
    {
      // Compute the scale contribution
      double comp1 = 1.0/(sqrt(2.0*PI)*this->DiffTransitionScaleSigma1)*exp(-0.5*pow((scaleDiff - this->DiffTransitionScaleMu1)/this->DiffTransitionScaleSigma1, 2.0));
      double comp2 = 1.0/(sqrt(2.0*PI)*this->DiffTransitionScaleSigma2)*exp(-0.5*pow((scaleDiff - this->DiffTransitionScaleMu2)/this->DiffTransitionScaleSigma2, 2.0));
      likelihood = this->DiffTransitionScaleWeight1*comp1 + this->DiffTransitionScaleWeight2*comp2;

      // Compute the angle contribution
      if ( angle < 20 )
	{
	  likelihood *= this->DiffTransitionAngleSlope1*angle + this->DiffTransitionAngleIntercept1;
	}
      else
	{
	  likelihood *= this->DiffTransitionAngleSlope2*angle + this->DiffTransitionAngleIntercept2;
	}
    }

  double tmp = likelihood*prior;

  return tmp;
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
      float tmpUpToDate   = 0.0;

      trellisGraph->GetVertexData()->GetArray( "particleIDArray" )->InsertTuple( id, &tmpParticleID );
      trellisGraph->GetVertexData()->GetArray( "stateArray" )->InsertTuple( id, &tmpState );
      trellisGraph->GetVertexData()->GetArray( "accumulatedWeightArray" )->InsertTuple( id, &tmpWeight );     
      trellisGraph->GetVertexData()->GetArray( "upToDateForForwardSearchArray" )->InsertTuple( id, &tmpUpToDate ); 

      (*graphNodeVisited)[subgraphID] = true;
    }

  // Now create directed edges from all newly created nodes to the 
  // nodes in the group passed to this function
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

  // Now loop over all the edges emanating from this node. If
  // a node in the subgraph hasn't been visited, it is a child
  // node of the current node and needs to be "expanded" in
  // the trellis graph
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
