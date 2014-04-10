/** \file
 *  \ingroup commandLineTools 
 *  \details This program ...
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipConventions.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipNelderMeadSimplexOptimizer.h"
#include "cipLobeSurfaceModelIO.h"
#include "FitLobeSurfaceModelsToParticleDataCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric* leftMetric = 
    new cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric();
    leftMetric->SetFissureSigmaDistance( fissureSigmaDistance );
    leftMetric->SetFissureSigmaTheta( fissureSigmaTheta );
    leftMetric->SetVesselSigmaDistance( vesselSigmaDistance );
    leftMetric->SetVesselSigmaTheta( vesselSigmaTheta );
    leftMetric->SetAirwaySigmaDistance( airwaySigmaDistance );
    leftMetric->SetAirwaySigmaTheta( airwaySigmaTheta );

  cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric* rightMetric = 
    new cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric();
    rightMetric->SetFissureSigmaDistance( fissureSigmaDistance );
    rightMetric->SetFissureSigmaTheta( fissureSigmaTheta );
    rightMetric->SetVesselSigmaDistance( vesselSigmaDistance );
    rightMetric->SetVesselSigmaTheta( vesselSigmaTheta );
    rightMetric->SetAirwaySigmaDistance( airwaySigmaDistance );
    rightMetric->SetAirwaySigmaTheta( airwaySigmaTheta );

  if ( leftFissureParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left lung fissure particles..." << std::endl;
      vtkPolyDataReader* leftFissureParticlesReader = vtkPolyDataReader::New();
        leftFissureParticlesReader->SetFileName( leftFissureParticlesFileName.c_str() );
	leftFissureParticlesReader->Update();    

    leftMetric->SetFissureParticles( leftFissureParticlesReader->GetOutput() );
    }
  if ( leftVesselParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left lung vessel particles..." << std::endl;
      vtkPolyDataReader* leftVesselParticlesReader = vtkPolyDataReader::New();
        leftVesselParticlesReader->SetFileName( leftVesselParticlesFileName.c_str() );
	leftVesselParticlesReader->Update();    

    leftMetric->SetVesselParticles( leftVesselParticlesReader->GetOutput() );
    }
  if ( leftAirwayParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left lung airway particles..." << std::endl;
      vtkPolyDataReader* leftAirwayParticlesReader = vtkPolyDataReader::New();
        leftAirwayParticlesReader->SetFileName( leftAirwayParticlesFileName.c_str() );
	leftAirwayParticlesReader->Update();    

    leftMetric->SetAirwayParticles( leftAirwayParticlesReader->GetOutput() );
    }

  if ( rightFissureParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading right lung fissure particles..." << std::endl;
      vtkPolyDataReader* rightFissureParticlesReader = vtkPolyDataReader::New();
        rightFissureParticlesReader->SetFileName( rightFissureParticlesFileName.c_str() );
	rightFissureParticlesReader->Update();    

    rightMetric->SetFissureParticles( rightFissureParticlesReader->GetOutput() );
    }
  if ( rightVesselParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading right lung vessel particles..." << std::endl;
      vtkPolyDataReader* rightVesselParticlesReader = vtkPolyDataReader::New();
        rightVesselParticlesReader->SetFileName( rightVesselParticlesFileName.c_str() );
	rightVesselParticlesReader->Update();    

    rightMetric->SetVesselParticles( rightVesselParticlesReader->GetOutput() );
    }
  if ( rightAirwayParticlesFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading right lung airway particles..." << std::endl;
      vtkPolyDataReader* rightAirwayParticlesReader = vtkPolyDataReader::New();
        rightAirwayParticlesReader->SetFileName( rightAirwayParticlesFileName.c_str() );
	rightAirwayParticlesReader->Update();    

    rightMetric->SetAirwayParticles( rightAirwayParticlesReader->GetOutput() );
    }

  // Read the left surface model
  unsigned int numberLeftModesUsed   = 0;
  double       leftWeightAccumulator = 0.0;
  cip::LobeSurfaceModelIO* leftModelIO = new cip::LobeSurfaceModelIO();
  if ( inLeftModelFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading left surface model..." << std::endl;
      leftModelIO->SetFileName( inLeftModelFileName );
      leftModelIO->Read();
      leftMetric->SetMeanSurfacePoints( leftModelIO->GetOutput()->GetMeanSurfacePoints() );

      while ( leftWeightAccumulator < 0.9 )
	{      
	  leftMetric->SetEigenvectorAndEigenvalue( &(*leftModelIO->GetOutput()->GetEigenvectors())[numberLeftModesUsed],       
						   (*leftModelIO->GetOutput()->GetEigenvalues())[numberLeftModesUsed] );      
	  leftWeightAccumulator += (*leftModelIO->GetOutput()->GetEigenvalues())[numberLeftModesUsed]/
	    leftModelIO->GetOutput()->GetEigenvalueSum();

	  numberLeftModesUsed++;
	}
      std::cout << "Fraction variance explained by modes used in left surface model:\t" << leftWeightAccumulator << std::endl;
    }

  // Read the right surface model
  unsigned int numberRightModesUsed   = 0;
  double       rightWeightAccumulator = 0.0;
  cip::LobeSurfaceModelIO* rightModelIO = new cip::LobeSurfaceModelIO();
  if ( inRightModelFileName.compare( "NA" ) != 0 )
    {
      std::cout << "Reading right surface model..." << std::endl;
      rightModelIO->SetFileName( inRightModelFileName );
      rightModelIO->Read();

      rightMetric->SetMeanSurfacePoints( rightModelIO->GetOutput()->GetMeanSurfacePoints() );

      while ( rightWeightAccumulator < 0.9 )
	{      
	  rightMetric->SetEigenvectorAndEigenvalue( &(*rightModelIO->GetOutput()->GetEigenvectors())[numberRightModesUsed],       
						   (*rightModelIO->GetOutput()->GetEigenvalues())[numberRightModesUsed] );      
	  rightWeightAccumulator += (*rightModelIO->GetOutput()->GetEigenvalues())[numberRightModesUsed]/
	    rightModelIO->GetOutput()->GetEigenvalueSum();
  
	  numberRightModesUsed++;
	}
      std::cout << "Fraction variance explained by modes used in right surface model:\t" << rightWeightAccumulator << std::endl;
    }

  // Now optimize the left surface model to fit the provided particles.
  // Optionally write the resulting, fittted model to file
  if ( inLeftModelFileName.compare( "NA" ) != 0 )
    {
      double* leftInitialParameters = new double[numberLeftModesUsed];
      double* leftOptimalParameters = new double[numberLeftModesUsed];
      for ( unsigned int i=0; i<numberLeftModesUsed; i++ )
	{
	  if ( useLeftModeWeights )
	    {
	      leftInitialParameters[i] = (*leftModelIO->GetOutput()->GetModeWeights())[i];
	    }
	  else
	    {
	      leftInitialParameters[i] = 0.0;
	    }
	}

      std::cout << "Executing Nelder-Mead optimizer (left lung)..." << std::endl;
      cipNelderMeadSimplexOptimizer* leftOptimizer = new cipNelderMeadSimplexOptimizer( numberLeftModesUsed );
	std::cout << "a" << std::endl;
        leftOptimizer->SetInitialParameters( leftInitialParameters );
	std::cout << "b" << std::endl;
	leftOptimizer->SetMetric( leftMetric );
	std::cout << "c" << std::endl;
	leftOptimizer->SetNumberOfIterations( numIters );
	std::cout << "d" << std::endl;
	leftOptimizer->Update();
	std::cout << "e" << std::endl;
	leftOptimizer->GetOptimalParameters( leftOptimalParameters );
	std::cout << "1" << std::endl;
      for ( unsigned int i=0; i<numberLeftModesUsed; i++ )
	{
	  (*leftModelIO->GetOutput()->GetModeWeights())[i] = leftOptimalParameters[i];
	}
	std::cout << "2" << std::endl;
      if ( outLeftModelFileName.compare( "NA" ) != 0 )
	{
	  std::cout << "Writing shape model to file (left lung)..." << std::endl;
	  leftModelIO->SetFileName( outLeftModelFileName );
	  leftModelIO->Write();
	}
	std::cout << "3" << std::endl;
    }

  // Now optimize the right surface model to fit the provided particles.
  // Optionally write the resulting, fittted model to file
  if ( inRightModelFileName.compare( "NA" ) != 0 )
    {
      double* rightInitialParameters = new double[numberRightModesUsed];
      double* rightOptimalParameters = new double[numberRightModesUsed];
      for ( unsigned int i=0; i<numberRightModesUsed; i++ )
	{
	  if ( useRightModeWeights )
	    {
	      rightInitialParameters[i] = (*rightModelIO->GetOutput()->GetModeWeights())[i];
	    }
	  else
	    {
	      rightInitialParameters[i] = 0.0;
	    }
	}

      std::cout << "Executing Nelder-Mead optimizer (right lung)..." << std::endl;
      cipNelderMeadSimplexOptimizer* rightOptimizer = new cipNelderMeadSimplexOptimizer( numberRightModesUsed );
        rightOptimizer->SetInitialParameters( rightInitialParameters );
	rightOptimizer->SetMetric( rightMetric );
	rightOptimizer->SetNumberOfIterations( numIters );
	rightOptimizer->Update();
	rightOptimizer->GetOptimalParameters( rightOptimalParameters );

      for ( unsigned int i=0; i<numberRightModesUsed; i++ )
	{
	  (*rightModelIO->GetOutput()->GetModeWeights())[i] = rightOptimalParameters[i];
	}

      if ( outRightModelFileName.compare( "NA" ) != 0 )
	{
	  std::cout << "Writing shape model to file (right lung)..." << std::endl;
	  rightModelIO->SetFileName( outRightModelFileName );
	  rightModelIO->Write();
	}
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
