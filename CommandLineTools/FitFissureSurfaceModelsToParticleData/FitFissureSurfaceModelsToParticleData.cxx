/** \file
 *  \ingroup commandLineTools 
 *  \details This program fits two TPS surface models to the same particles data
 *  set in an iterative fashion. It is meant to be used for fitting the
 *  right oblique and the right horizontal surface models. The idea is
 *  to optimize each surface location a little, and then reweight the
 *  particles based on how much they agree with the current TPS surface
 *  locations. So the RO model will gravitate towards more likely RO
 *  particles, and the RH model will graviate towards more likely RH
 *  model 
 *
 *  USAGE: 
 *
 *  FitFissureSurfaceModelsToParticleData  [-n \<unsigned int\>] [-w \<bool\>]
 *                                         [-t \<double\>] [-d \<double\>] 
 *                                         [-v \<double\>] --outRHModel \<string\>
 *                                         --inRHModel \<string\> --outROModel
 *                                         \<string\> --inROModel \<string\> -p
 *                                         \<string\> [--] [--version] [-h]
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
 *   --outRHModel \<string\>
 *     (required)  Output right horizontal shape model file name
 *
 *   --inRHModel \<string\>
 *     (required)  Input right horizontal shape model file name
 *
 *   --outROModel \<string\>
 *     (required)  Output right oblique shape model file name
 *
 *   --inROModel \<string\>
 *     (required)  Input right oblique shape model file name
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
 *  $Date: 2012-09-05 21:23:51 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 240 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipConventions.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipNelderMeadSimplexOptimizer.h"
#include "cipLobeBoundaryShapeModelIO.h"
#include "cipLobeBoundaryShapeModel.h"
#include "FitFissureSurfaceModelsToParticleDataCLP.h"

unsigned int DetermineNumberOfModesToUse( cipLobeBoundaryShapeModel*, double );
void GetOptimalParametersFromOptimization( cipLobeBoundaryShapeModel*, vtkPolyData*, std::vector< double >*, unsigned int, double*, double*,
                                           double, double, unsigned int );
void GetAngleAndDistanceFromThinPlateSplineSurface( double*, double*, cipThinPlateSplineSurface*, double*, double* );
double GetVectorMagnitude( double[3] );
double GetAngleBetweenVectors( double[3], double[3] );
void UpdateParticleWeights( vtkPolyData*, std::vector< double >*, std::vector< double >*, double*, double*, 
                            cipLobeBoundaryShapeModel*, cipLobeBoundaryShapeModel*, double, double, unsigned int, unsigned int );

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "Reading polydata..." << std::endl;
  vtkPolyDataReader* particlesReader = vtkPolyDataReader::New();
    particlesReader->SetFileName( particlesFileName.c_str() );
    particlesReader->Update();    

  // Read shape models
  std::cout << "Reading RO shape model..." << std::endl;
  cipLobeBoundaryShapeModelIO* roModelIO = new cipLobeBoundaryShapeModelIO();
    roModelIO->SetFileName( inROShapeModelFileName );
    roModelIO->Read();

  std::cout << "Reading RH shape model..." << std::endl;
  cipLobeBoundaryShapeModelIO* rhModelIO = new cipLobeBoundaryShapeModelIO();
    rhModelIO->SetFileName( inRHShapeModelFileName );
    rhModelIO->Read();

  // Initialize the initial parameters for the RO and RH. First
  // determine how many modes to use for each
  unsigned int numberModesRO = DetermineNumberOfModesToUse( roModelIO->GetOutput(), shapeVarianceThreshold );
  unsigned int numberModesRH = DetermineNumberOfModesToUse( rhModelIO->GetOutput(), shapeVarianceThreshold );

  std::cout << "Using " << numberModesRO << " modes for the right oblique model..." << std::endl;
  std::cout << "Using " << numberModesRH << " modes for the right horizontal model..." << std::endl;

  double* parametersRO = new double[numberModesRO];
  for ( unsigned int i=0; i<numberModesRO; i++ )
    {
    if ( useModeWeights )
      {
	parametersRO[i] = (*roModelIO->GetOutput()->GetModeWeights())[i];
      }
    else
      {
	parametersRO[i] = 0.0;
      }
    }

  double* parametersRH = new double[numberModesRH];
  for ( unsigned int i=0; i<numberModesRH; i++ )
    {
    if ( useModeWeights )
      {
	parametersRH[i] = (*rhModelIO->GetOutput()->GetModeWeights())[i];
      }
    else
      {
	parametersRH[i] = 0.0;
      }
    }

  // Initialize the particle weights
  std::cout << "Initializing particle weights..." << std::endl;
  std::vector< double > roParticleWeights;
  std::vector< double > rhParticleWeights;

  for ( unsigned int i=0; i<particlesReader->GetOutput()->GetNumberOfPoints(); i++ )
    {
    roParticleWeights.push_back( 1.0 );
    rhParticleWeights.push_back( 1.0 );
    }

  // Now enter the main optimization loop  
  for ( unsigned int i=0; i<2; i++ )
    {
    std::cout << "Updating particle weights..." << std::endl;
    UpdateParticleWeights( particlesReader->GetOutput(), &roParticleWeights, &rhParticleWeights, 
                           parametersRO, parametersRH, roModelIO->GetOutput(), rhModelIO->GetOutput(), sigmaDistance, sigmaTheta,
                           numberModesRO, numberModesRH );

    std::cout << "Optimizing RO..." << std::endl;
    GetOptimalParametersFromOptimization( roModelIO->GetOutput(), particlesReader->GetOutput(), &roParticleWeights,
                                          50, parametersRO, parametersRO, sigmaDistance, sigmaTheta, numberModesRO );

    std::cout << "Updating particle weights..." << std::endl;
    UpdateParticleWeights( particlesReader->GetOutput(), &roParticleWeights, &rhParticleWeights, 
                           parametersRO, parametersRH, roModelIO->GetOutput(), rhModelIO->GetOutput(), sigmaDistance, sigmaTheta,
                           numberModesRO, numberModesRH );

    std::cout << "Optimizing RH..." << std::endl;
    GetOptimalParametersFromOptimization( rhModelIO->GetOutput(), particlesReader->GetOutput(), &rhParticleWeights,
                                          50, parametersRH, parametersRH, sigmaDistance, sigmaTheta, numberModesRH );
    }
  
  // Now create and write the final models
  for ( unsigned int i=0; i<numberModesRO; i++ )
    {
      (*roModelIO->GetOutput()->GetModeWeights())[i] = parametersRO[i];
    }

  std::cout << "Writing RO shape model to file..." << std::endl;
  roModelIO->SetFileName( outROShapeModelFileName );
  roModelIO->Write();

  for ( unsigned int i=0; i<numberModesRH; i++ )
    {
      (*rhModelIO->GetOutput()->GetModeWeights())[i] = parametersRH[i];
    }

  std::cout << "Writing RH shape model to file..." << std::endl;
  rhModelIO->SetFileName( outRHShapeModelFileName );
  rhModelIO->Write();

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}


void GetOptimalParametersFromOptimization( cipLobeBoundaryShapeModel* model, vtkPolyData* particles, std::vector< double >* particleWeights,
                                           unsigned int numIters, double* initialParameters, double* optimalParameters,
                                           double sigmaDistance, double sigmaTheta, unsigned int numberModesToUse )
{
  cipThinPlateSplineSurfaceModelToParticlesMetric tpsToParticlesMetric;
    tpsToParticlesMetric.SetMeanSurfacePoints( model->GetMeanSurfacePoints() );
    tpsToParticlesMetric.SetParticles( particles );
    tpsToParticlesMetric.SetSigmaDistance( sigmaDistance );
    tpsToParticlesMetric.SetSigmaTheta( sigmaTheta );
    tpsToParticlesMetric.SetParticleWeights( particleWeights );
  for ( unsigned int i=0; i<numberModesToUse; i++ )
    {
      tpsToParticlesMetric.SetEigenvectorAndEigenvalue( &(*model->GetEigenvectors())[i], 
							(*model->GetEigenvalues())[i] );  
    }

  std::cout << "Executing Nelder Mead Optimizer..." << std::endl;
  cipNelderMeadSimplexOptimizer* nelderMeadOptimizer = new cipNelderMeadSimplexOptimizer( numberModesToUse );
    nelderMeadOptimizer->SetInitialParameters( initialParameters );
    nelderMeadOptimizer->SetMetric( &tpsToParticlesMetric );
    nelderMeadOptimizer->SetNumberOfIterations( numIters );
    nelderMeadOptimizer->Update();
    nelderMeadOptimizer->GetOptimalParameters( optimalParameters );
}


unsigned int DetermineNumberOfModesToUse( cipLobeBoundaryShapeModel* model, double shapeVarianceThreshold )
{
  double weightAccumulator = 0.0;
  unsigned int numberModesToUse = 0;
  while ( weightAccumulator <= shapeVarianceThreshold && numberModesToUse < model->GetNumberOfModes() && numberModesToUse < 9 )
    {
      weightAccumulator += (*model->GetEigenvalues())[numberModesToUse]/static_cast< double >( model->GetEigenvalueSum() );
  
      numberModesToUse++;
    }

  return numberModesToUse;
}


void UpdateParticleWeights( vtkPolyData* particles, std::vector< double >* roParticleWeights, std::vector< double >*  rhParticleWeights, 
                            double* parametersRO, double* parametersRH, cipLobeBoundaryShapeModel* roModel, cipLobeBoundaryShapeModel* rhModel,
                            double sigmaDistance, double sigmaTheta, unsigned int numberModesRO, unsigned int numberModesRH )
{
  double distanceRO, distanceRH;
  double angleRO, angleRH;

  double valueRO, valueRH;

  bool addPoint;

  //
  // Create the RO TPS
  //
  std::vector< double* > surfacePointsRO;

  for ( unsigned int i=0; i<(*roModel->GetMeanSurfacePoints()).size(); i++ )
    {
    addPoint = true;

    double* point = new double[3];
      point[0] = (*roModel->GetMeanSurfacePoints())[i][0];
      point[1] = (*roModel->GetMeanSurfacePoints())[i][1];
      point[2] = (*roModel->GetMeanSurfacePoints())[i][2];
    
    for ( unsigned int m=0; m<numberModesRO; m++ )
      {		
      point[2] += parametersRO[m]*vcl_sqrt( (*roModel->GetEigenvalues())[m] )*(*roModel->GetEigenvectors())[m][i];
      }

    for ( unsigned int j=0; j<surfacePointsRO.size(); j++ )
      {
      if ( surfacePointsRO[j][0] == point[0] && surfacePointsRO[j][1] == point[1] )
        {
        addPoint = false;
        }
      }

    if ( addPoint )
      {
      surfacePointsRO.push_back( point );
      }
    }

  cipThinPlateSplineSurface* tpsRO = new cipThinPlateSplineSurface();
    tpsRO->SetSurfacePoints( &surfacePointsRO );

  //
  // Create the RH TPS
  //
  std::vector< double* > surfacePointsRH;

  for ( unsigned int i=0; i<(*rhModel->GetMeanSurfacePoints()).size(); i++ )
    {
    addPoint = true;
    
    double* point = new double[3];
      point[0] = (*rhModel->GetMeanSurfacePoints())[i][0];
      point[1] = (*rhModel->GetMeanSurfacePoints())[i][1];
      point[2] = (*rhModel->GetMeanSurfacePoints())[i][2];

    for ( unsigned int m=0; m<numberModesRH; m++ )
      {
	point[2] += parametersRH[m]*vcl_sqrt( (*rhModel->GetEigenvalues())[m] )*(*rhModel->GetEigenvectors())[m][i];
      }

    for ( unsigned int j=0; j<surfacePointsRH.size(); j++ )
      {
      if ( surfacePointsRH[j][0] == point[0] && surfacePointsRH[j][1] == point[1] )
        {
        addPoint = false;
        }
      }

    if ( addPoint )
      {
      surfacePointsRH.push_back( point );
      }
    }

  cipThinPlateSplineSurface* tpsRH = new cipThinPlateSplineSurface();
    tpsRH->SetSurfacePoints( &surfacePointsRH );

  //
  // Now loop through the particles
  //
  double* particleOrientation = new double[3];
  double* particlePosition    = new double[3];

  double rhHeight, roHeight;

  for ( unsigned int i=0; i<particles->GetNumberOfPoints(); i++ )
    {
    particlePosition[0] = particles->GetPoint(i)[0];
    particlePosition[1] = particles->GetPoint(i)[1];
    particlePosition[2] = particles->GetPoint(i)[2];

    rhHeight = tpsRH->GetSurfaceHeight( particlePosition[0], particlePosition[1] );
    roHeight = tpsRO->GetSurfaceHeight( particlePosition[0], particlePosition[1] );

    if ( roHeight > rhHeight )
      {
      (*roParticleWeights)[i] = 1.0;
      (*rhParticleWeights)[i] = 0.0;
      }
    else
      {
      particleOrientation[0] = particles->GetFieldData()->GetArray( "hevec2" )->GetTuple(i)[0];
      particleOrientation[1] = particles->GetFieldData()->GetArray( "hevec2" )->GetTuple(i)[1];
      particleOrientation[2] = particles->GetFieldData()->GetArray( "hevec2" )->GetTuple(i)[2];
    
      GetAngleAndDistanceFromThinPlateSplineSurface( particlePosition, particleOrientation, tpsRO, &distanceRO, &angleRO );
      GetAngleAndDistanceFromThinPlateSplineSurface( particlePosition, particleOrientation, tpsRH, &distanceRH, &angleRH );

      valueRO = std::exp( -0.5*std::pow(distanceRO/sigmaDistance,2) )*std::exp( -0.5*std::pow(angleRO/sigmaTheta,2) );    
      valueRH = std::exp( -0.5*std::pow(distanceRH/sigmaDistance,2) )*std::exp( -0.5*std::pow(angleRH/sigmaTheta,2) );

      (*roParticleWeights)[i] = valueRO/(valueRO+valueRH);
      (*rhParticleWeights)[i] = valueRH/(valueRO+valueRH);
      }
    }

  delete[] particleOrientation;
  delete[] particlePosition;
}


void GetAngleAndDistanceFromThinPlateSplineSurface( double* position, double* orientation, cipThinPlateSplineSurface* tps,
                                                    double* distance, double* angle )
{
  double* normal = new double[3];

  cipParticleToThinPlateSplineSurfaceMetric* particleToTPSMetric = new cipParticleToThinPlateSplineSurfaceMetric();
    particleToTPSMetric->SetThinPlateSplineSurface( tps );

  cipNewtonOptimizer< 2 >* optimizer = new cipNewtonOptimizer< 2 >();
    optimizer->SetMetric( particleToTPSMetric );

  cipNewtonOptimizer< 2 >::PointType* domainParams  = new cipNewtonOptimizer< 2 >::PointType( 2 );
  cipNewtonOptimizer< 2 >::PointType* optimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  //
  // Determine the domain location for which the particle is closest
  // to the TPS surface
  //
  particleToTPSMetric->SetParticle( position );

  //
  // The particle's x, and y location are a good place to initialize
  // the search for the domain locations that result in the smallest
  // distance between the particle and the TPS surface
  //
  (*domainParams)[0] = position[0]; 
  (*domainParams)[1] = position[1]; 

  //
  // Perform Newton line search to determine the closest point on
  // the current TPS surface
  //
  optimizer->SetInitialParameters( domainParams );
  optimizer->Update();

  tps->GetSurfaceNormal( (*optimalParams)[0], (*optimalParams)[1], normal );
  *angle = GetAngleBetweenVectors( normal, orientation );

  //
  // Get the distance between the particle and the TPS surface. This
  // is just the square root of the objective function value
  // optimized by the Newton method.
  //
  *distance = vcl_sqrt( optimizer->GetOptimalValue() );

  delete optimizer;

  delete[] normal;
}


double GetAngleBetweenVectors( double vec1[3], double vec2[3] )
{
  double vec1Mag = GetVectorMagnitude( vec1 );
  double vec2Mag = GetVectorMagnitude( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( vcl_abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = 180.0*vcl_acos( arg )/vnl_math::pi;

  if ( angle > 90.0 )
    {
    angle = 180.0 - angle;
    }

  return angle;   
}


double GetVectorMagnitude( double vector[3] )
{
  double magnitude = vcl_sqrt( std::pow( vector[0], 2 ) + std::pow( vector[1], 2 ) + std::pow( vector[2], 2 ) );

  return magnitude;
}

#endif
