/** \file
 *  \ingroup commandLineTools 
 *  \details This program accepts as input a VTK polydata file
 *  corresponding to airway particles data and produces a VTK polydata
 *  file containing filtered particles labeled by generation.
 * 
 * Impose topology / treating each particle as a graph node, first
 * determine connections between nodes, then determine direction of
 * flow. Graph is assumed to be acyclic, but we do not assume that it
 * is connected. The framework allows for disconnected subgraphs to
 * exist (providing for the case in which we see airway obstruction,
 * due to mucous plugs, e.g., that 
 *
 * This program accepts as input a VTK polydata file corresponding to
 * airway particles data and produces a VTK polydata file containing
 * filtered particles labeled by generation. The input particles are
 * first processed by imposing a topology on the particles point
 * set. This topology is represented by a graph bidirectional edges
 * are formed between a pair of particles provided the pair are both
 * spatially close to one another and sufficiently aligned, where
 * alignment is defined according to the vector connecting the two
 * particles and the angles formed between that vector and the
 * particles' minor eigenvectors. (The user can specify the spatial
 * distance threshold using the -pdt flag and the angle threshold
 * using the -pat flag). Both the distance and angle thresholds are
 * learned from training data. 
 *
 * Training data / kernel density estimation / HMM\n";
 *
 *
 *
 *  Usage: FilterAndLabelAirwayParticlesByGeneration \<options\> where \<options\> is one or more of the following:\n
 *    \<-h\>     Display (this) usage information\n
 *    \<-i\>     Input airway particles file name\n
 *    \<-o\>     Output airway particles file name\n
 */


#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkSmartPointer.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.h"
#include "itkIdentityTransform.h"
#include "cipChestConventions.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include <cfloat>
#include <math.h>
#include <fstream>
#include "LabelAirwayParticlesByGenerationCLP.h"

void PrintResults( vtkSmartPointer< vtkPolyData >, vtkSmartPointer< vtkPolyData > );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Read the particles to which generation labels are to be assigned
  // std::cout << "Reading airway particles..." << std::endl;
  // vtkSmartPointer< vtkPolyDataReader > particlesReader = vtkSmartPointer< vtkPolyDataReader >::New();
  //   particlesReader->SetFileName( inParticlesFileName.c_str() );
  //   particlesReader->Update();

  // vtkSmartPointer< vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter > particlesToGenLabeledParticles = 
  //   vtkSmartPointer< vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter >::New();
  //   particlesToGenLabeledParticles->SetInput( particlesReader->GetOutput() );
  //   particlesToGenLabeledParticles->SetParticleDistanceThreshold( particleDistanceThreshold );
  //   particlesToGenLabeledParticles->SetKernelDensityEstimationROIRadius( kernelDensityEstimationROIRadius );
  //   if ( kdeMode )
  //     {
  // 	particlesToGenLabeledParticles->SetModeToKDE();
  //     }
  //   if ( particleRoot >= 0 )
  //     {
  // 	particlesToGenLabeledParticles->SetParticleRootNodeID( particleRoot );
  //     }

  // for ( unsigned int i=0; i<airwayGenerationLabeledAtlasFileNames.size(); i++ )
  //   {
  //   std::cout << "Reading atlas..." << std::endl;
  //   vtkSmartPointer< vtkPolyDataReader > atlasReader = vtkSmartPointer< vtkPolyDataReader >::New();
  //     atlasReader->SetFileName( airwayGenerationLabeledAtlasFileNames[i].c_str() );
  //     atlasReader->Update();

  //   particlesToGenLabeledParticles->AddAirwayGenerationLabeledAtlas( atlasReader->GetOutput() );
  //   }
  // particlesToGenLabeledParticles->Update();

  // std::cout << "Writing generation-labeled airway particles..." << std::endl;
  // vtkSmartPointer< vtkPolyDataWriter > particlesWriter = vtkSmartPointer< vtkPolyDataWriter >::New();
  //   particlesWriter->SetFileName( outParticlesFileName.c_str() );
  //   particlesWriter->SetInput( particlesToGenLabeledParticles->GetOutput() ); 
  //   particlesWriter->Update();

  // // Optionally print results
  // if ( printResults )
  //   {
  //     PrintResults( particlesReader->GetOutput(), particlesToGenLabeledParticles->GetOutput() );
  //   }

  // std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

void PrintResults( vtkSmartPointer< vtkPolyData > testParticles, vtkSmartPointer< vtkPolyData > refParticles )
{
  std::vector< unsigned char > statesVec;
  statesVec.push_back( (unsigned char)( cip::TRACHEA ) );
  statesVec.push_back( (unsigned char)( cip::MAINBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::UPPERLOBEBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::SUPERIORDIVISIONBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::LINGULARBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::MIDDLELOBEBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::INTERMEDIATEBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::LOWERLOBEBRONCHUS ) );
  statesVec.push_back( (unsigned char)( cip::AIRWAYGENERATION3 ) );
  statesVec.push_back( (unsigned char)( cip::AIRWAYGENERATION4 ) );
  statesVec.push_back( (unsigned char)( cip::AIRWAYGENERATION5 ) );

  // Create a confusion matrix
  std::vector< std::vector< unsigned int > > confusionMatrix;      
  for ( unsigned int i=0; i<statesVec.size(); i++ )
    {
      std::vector< unsigned int > tmp;
      for ( unsigned int j=0; j<statesVec.size(); j++ )
	{
	  tmp.push_back( 0 );
	}
      confusionMatrix.push_back(tmp);
    }
      
  std::map< unsigned char, unsigned int > intersectionCounter;
  std::map< unsigned char, unsigned int > inTypeCounter;
  std::map< unsigned char, unsigned int > outTypeCounter;
      
  for ( unsigned int i=0; i<statesVec.size(); i++ )
    {
      intersectionCounter[statesVec[i]] = 0;
      inTypeCounter[statesVec[i]]       = 0;
      outTypeCounter[statesVec[i]]      = 0;
    }

  unsigned int trueStateIndex  = 0;
  unsigned int guessStateIndex = 0;
  unsigned char inType, outType;
  for ( unsigned int p=0; p<refParticles->GetNumberOfPoints(); p++ )
    {
      inType  = (unsigned char)( refParticles->GetPointData()->GetArray( "ChestType" )->GetTuple(p)[0] );
      outType = (unsigned char)( testParticles->GetPointData()->GetArray( "ChestType" )->GetTuple(p)[0] );

      for ( unsigned int i=0; i<statesVec.size(); i++ )
	{	      
	  if ( int(statesVec[i]) == int(inType) )
	    {
	      trueStateIndex = i;
	    }
	  if ( int(statesVec[i]) == int(outType) )
	    {
	      guessStateIndex = i;
	    }
	}
      confusionMatrix[trueStateIndex][guessStateIndex] += 1;
      
      if ( inType == outType )
	{
	  intersectionCounter[inType]++;
	}
      inTypeCounter[inType]++;
      outTypeCounter[outType]++;
    }
  
  cip::ChestConventions conventions;
  for ( unsigned int i=0; i<statesVec.size(); i++ )
    {
      double num    = static_cast< float >( intersectionCounter[statesVec[i]] );
      double denom1 = static_cast< float >( inTypeCounter[statesVec[i]] );
      double denom2 = static_cast< float >( outTypeCounter[statesVec[i]] );
      double denom  = denom1 + denom2;
      
      if ( denom > 0.0 )
	{
	  std::cout << "Dice for " << conventions.GetChestTypeName( statesVec[i] ) << ":\t" << 2.0*num/(denom ) << std::endl;
	}
    }
  
  std::cout << "----------------- Confusion Matrix -----------------------" << std::endl;
  for ( unsigned int i=0; i<statesVec.size(); i++ )
    {
      for ( unsigned int j=0; j<statesVec.size(); j++ )
	{
	  std::cout << confusionMatrix[i][j] << "\t";
	}
      std::cout << std::endl;
    }
}

#endif
