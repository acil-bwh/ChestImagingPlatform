/** \file
 *  \ingroup commandLineTools 
 *  \details This program is used to fit a left oblique shape 
 *  model to fissure particles data. It is expected that the input 
 *  shape model is generated with the 'GenerateFissureShapeModels'
 *  program. The output shape model has the same form as the input 
 *  shape model, but it indicates how much to weight each of the 
 *  primary modes of variation in order to achieve a good fit to the 
 *  particles data.
 *
 *  USAGE: 
 *
 *  FitFissureSurfaceModelToParticleData  [-n \<unsigned int\>] [-w \<bool\>]
 *                                        [-t \<double\>] [-d \<double\>] 
 *                                        [-v \<double\>] -o \<string\> -i \<string\>
 *                                        -p \<string\> [--] [--version] [-h]
 *
 *  Where: 
 *
 *   -n \<unsigned int\>,  --numIters \<unsigned int\>
 *     Number of iterations to perform for Nelder-Mead simplex model fitting.
 *
 *   -w \<bool\>,  --modeWeights \<bool\>
 *     Set to 1 to use stored mode weights for initialization. Set to 0
 *     otherwise (0 by default)
 *
 *   -t \<double\>,  --sigTheta \<double\>
 *     Sigma theta value for the TPS to particles optimization
 *
 *   -d \<double\>,  --sigDist \<double\>
 *     Sigma distance value for the TPS to particles optimization
 *
 *   -v \<double\>,  --shapeVar \<double\>
 *     Shape variance threshold. This indicates how much of the variance you
 *     want accounted for during the shape model fitting process.
 *
 *   -o \<string\>,  --outModel \<string\>
 *     (required)  Output shape model file name
 *
 *   -i \<string\>,  --inModel \<string\>
 *     (required)  Input shape model file name
 *
 *   -p \<string\>,  --inFile \<string\>
 *     (required)  Input particles file name (vtk)
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date: 2012-09-05 20:37:50 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 238 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipNelderMeadSimplexOptimizer.h"
#include "cipLobeBoundaryShapeModelIO.h"


int main( int argc, char *argv[] )
{
  //
  // Define arguments
  //
  std::string particlesFileName        = "NA";
  std::string inShapeModelFileName     = "NA";
  std::string outShapeModelFileName    = "NA";
  double shapeVarianceThreshold        = 0.95;
  double sigmaDistance                 = 20.0;
  double sigmaTheta                    = 20.0;
  bool   useModeWeights                = 0;
  unsigned int numIters                = 0;

  //
  // Program and argument descriptions for user help
  //
  std::string programDesc = "This program is used to fit a left oblique shape model to fissure \
particles data. It is expected that the input shape model is generated with the 'GenerateFissureShapeModels' \
program. The output shape model has the same form as the input shape model, but it indicates how much to \
weight each of the primary modes of variation in order to achieve a good fit to the particles data.";

  std::string particlesFileNameDesc = "Input particles file name (vtk)";
  std::string inShapeModelFileNameDesc = "Input shape model file name";
  std::string outShapeModelFileNameDesc = "Output shape model file name";
  std::string shapeVarianceThresholdDesc = "Shape variance threshold. This indicates how much of \
the variance you want accounted for during the shape model fitting process.";
  std::string sigmaDistanceDesc = "Sigma distance value for the TPS to particles optimization";
  std::string sigmaThetaDesc = "Sigma theta value for the TPS to particles optimization";
  std::string useModeWeightsDesc = "Set to 1 to use stored mode weights for initialization. Set to 0 otherwise (0 by default)";
  std::string numItersDesc = "Number of iterations to perform for Nelder-Mead simplex model fitting.";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 238 $" );

    TCLAP::ValueArg<std::string> particlesFileNameArg( "p", "inFile", particlesFileNameDesc, true, particlesFileName, "string", cl );
    TCLAP::ValueArg<std::string> inShapeModelFileNameArg( "i", "inModel", inShapeModelFileNameDesc, true, inShapeModelFileName, "string", cl );
    TCLAP::ValueArg<std::string> outShapeModelFileNameArg( "o", "outModel", outShapeModelFileNameDesc, true, outShapeModelFileName, "string", cl );
    TCLAP::ValueArg<double> shapeVarianceThresholdArg( "v", "shapeVar", shapeVarianceThresholdDesc, false, shapeVarianceThreshold, "double", cl );
    TCLAP::ValueArg<double> sigmaDistanceArg( "d", "sigDist", sigmaDistanceDesc, false, sigmaDistance, "double", cl );
    TCLAP::ValueArg<double> sigmaThetaArg( "t", "sigTheta", sigmaThetaDesc, false, sigmaTheta, "double", cl );
    TCLAP::ValueArg<bool> useModeWeightsArg( "w", "modeWeights", useModeWeightsDesc, false, useModeWeights, "bool", cl );
    TCLAP::ValueArg<unsigned int> numItersArg( "n", "numIters", numItersDesc, false, numIters, "unsigned int", cl );

    cl.parse( argc, argv );

    particlesFileName            = particlesFileNameArg.getValue();
    inShapeModelFileName         = inShapeModelFileNameArg.getValue();
    outShapeModelFileName        = outShapeModelFileNameArg.getValue();
    shapeVarianceThreshold       = shapeVarianceThresholdArg.getValue();
    sigmaDistance                = sigmaDistanceArg.getValue();
    sigmaTheta                   = sigmaThetaArg.getValue();
    useModeWeights               = useModeWeightsArg.getValue();
    numIters                     = numItersArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  std::cout << "Reading polydata..." << std::endl;
  vtkPolyDataReader* particlesReader = vtkPolyDataReader::New();
    particlesReader->SetFileName( particlesFileName.c_str() );
    particlesReader->Update();    

  //
  // Read shape model
  //
  std::cout << "Reading shape model..." << std::endl;
  cipLobeBoundaryShapeModelIO* modelIO = new cipLobeBoundaryShapeModelIO();
    modelIO->SetFileName( inShapeModelFileName );
    modelIO->Read();

  unsigned int numberModesUsed   = 0;
  double       weightAccumulator = 0.0;
  cipThinPlateSplineSurfaceModelToParticlesMetric* tpsToParticlesMetric = new cipThinPlateSplineSurfaceModelToParticlesMetric();
    tpsToParticlesMetric->SetMeanSurfacePoints( modelIO->GetOutput()->GetMeanSurfacePoints() );
    tpsToParticlesMetric->SetParticles( particlesReader->GetOutput() );
    tpsToParticlesMetric->SetSigmaDistance( sigmaDistance );
    tpsToParticlesMetric->SetSigmaTheta( sigmaTheta );
//  while ( weightAccumulator < shapeVarianceThreshold && numberModesUsed < 9 )
  while ( numberModesUsed < 9 )
    {      
    tpsToParticlesMetric->SetEigenvectorAndEigenvalue( &(*modelIO->GetOutput()->GetEigenvectors())[numberModesUsed],       
						       (*modelIO->GetOutput()->GetEigenvalues())[numberModesUsed] );      
    weightAccumulator += (*modelIO->GetOutput()->GetEigenvalues())[numberModesUsed]/
      modelIO->GetOutput()->GetEigenvalueSum();
  
    numberModesUsed++;
    }
  std::cout << "Variance explained:\t" << weightAccumulator << std::endl;

  double* initialParameters = new double[numberModesUsed];
  double* optimalParameters = new double[numberModesUsed];
  for ( unsigned int i=0; i<numberModesUsed; i++ )
    {
    if ( useModeWeights )
      {
	initialParameters[i] = (*modelIO->GetOutput()->GetModeWeights())[i];
      }
    else
      {
	initialParameters[i] = 0.0;
      }
    }

  std::cout << "Executing Nelder Mead Optimizer..." << std::endl;
  cipNelderMeadSimplexOptimizer* nelderMeadOptimizer = new cipNelderMeadSimplexOptimizer( numberModesUsed );
    nelderMeadOptimizer->SetInitialParameters( initialParameters );
    nelderMeadOptimizer->SetMetric( tpsToParticlesMetric );
    nelderMeadOptimizer->SetNumberOfIterations( numIters );
    nelderMeadOptimizer->Update();
    nelderMeadOptimizer->GetOptimalParameters( optimalParameters );

  for ( unsigned int i=0; i<numberModesUsed; i++ )
    {
      (*modelIO->GetOutput()->GetModeWeights())[i] = optimalParameters[i];
    }

  std::cout << "Writing shape model to file..." << std::endl;
  modelIO->SetFileName( outShapeModelFileName );
  modelIO->Write();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
