/**
 *
 *  $Date: 2012-09-17 18:32:23 -0400 (Mon, 17 Sep 2012) $
 *  $Revision: 268 $
 *  $Author: jross $
 *
 */

#ifndef __cipThinPlateSplineSurfaceModelToParticlesMetric_cxx
#define __cipThinPlateSplineSurfaceModelToParticlesMetric_cxx

#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "cipHelper.h"

cipThinPlateSplineSurfaceModelToParticlesMetric
::cipThinPlateSplineSurfaceModelToParticlesMetric()
{
  this->FissureParticles = vtkPolyData::New();
  this->AirwayParticles  = vtkPolyData::New();
  this->VesselParticles  = vtkPolyData::New();

  this->RegularizationWeight = 0.0;
  this->NumberOfModes = 0; 

  this->NumberOfFissureParticles = 0;
  this->NumberOfAirwayParticles  = 0;
  this->NumberOfVesselParticles  = 0;

  this->FissureTermWeight = 1.0;
  this->VesselTermWeight  = 1.0;
  this->AirwayTermWeight  = 1.0;
}


cipThinPlateSplineSurfaceModelToParticlesMetric
::~cipThinPlateSplineSurfaceModelToParticlesMetric()
{
  this->FissureParticles->Delete();
  this->AirwayParticles->Delete();
  this->VesselParticles->Delete();
}


void cipThinPlateSplineSurfaceModelToParticlesMetric::SetFissureParticles( vtkPolyData* const particles )
{
  this->FissureParticles = particles;
  this->NumberOfFissureParticles = this->FissureParticles->GetNumberOfPoints();

  // If no particle weights have already been specified, set each
  // particle to have equal, unity weight
  if ( this->FissureParticleWeights.size() == 0 )
    {
    for ( unsigned int i=0; i<this->NumberOfFissureParticles; i++ )
      {
      this->FissureParticleWeights.push_back( 1.0 );
      }
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetVesselParticles( vtkPolyData* const particles )
{
  this->VesselParticles = particles;
  this->NumberOfVesselParticles = this->VesselParticles->GetNumberOfPoints();

  // If no particle weights have already been specified, set each
  // particle to have equal, unity weight
  if ( this->VesselParticleWeights.size() == 0 )
    {
    for ( unsigned int i=0; i<this->NumberOfVesselParticles; i++ )
      {
      this->VesselParticleWeights.push_back( 1.0 );
      }
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetAirwayParticles( vtkPolyData* const particles )
{
  this->AirwayParticles = particles;
  this->NumberOfAirwayParticles = this->AirwayParticles->GetNumberOfPoints();

  // If no particle weights have already been specified, set each
  // particle to have equal, unity weight
  if ( this->AirwayParticleWeights.size() == 0 )
    {
    for ( unsigned int i=0; i<this->NumberOfAirwayParticles; i++ )
      {
      this->AirwayParticleWeights.push_back( 1.0 );
      }
    }
}


void cipThinPlateSplineSurfaceModelToParticlesMetric::SetFissureParticleWeights( std::vector< double >* weights )
{
  // Clear any existing particle weights first
  this->FissureParticleWeights.clear();

  for ( unsigned int i=0; i<weights->size(); i++ )
    {
    this->FissureParticleWeights.push_back( (*weights)[i] );
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetAirwayParticleWeights( std::vector< double >* weights )
{
  // Clear any existing particle weights first
  this->AirwayParticleWeights.clear();

  for ( unsigned int i=0; i<weights->size(); i++ )
    {
    this->AirwayParticleWeights.push_back( (*weights)[i] );
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetVesselParticleWeights( std::vector< double >* weights )
{
  // Clear any existing particle weights first
  this->VesselParticleWeights.clear();

  for ( unsigned int i=0; i<weights->size(); i++ )
    {
    this->VesselParticleWeights.push_back( (*weights)[i] );
    }
}


// Set the surface points of the mean surface model. These points --
// in conjunction with the various model eigenvectors and eigenvalues
// -- will be used to construct the TPS surfaces
void cipThinPlateSplineSurfaceModelToParticlesMetric::SetMeanSurfacePoints( const std::vector< cip::PointType >& meanPoints ) 
{
  this->NumberOfSurfacePoints = meanPoints.size();

  for ( unsigned int i=0; i<this->NumberOfSurfacePoints; i++ )
    {
      cip::PointType point(3);
        point[0] = meanPoints[i][0];
	point[1] = meanPoints[i][1];
	point[2] = meanPoints[i][2];

      this->MeanPoints.push_back( point );
    
      // We will also initialize 'SurfacePoints' here. Calls to the
      // 'GetValue' method will alter the z-values of 'SurfacePoints',
      // but the x and y locations will remain the same no matter what. 
      this->SurfacePoints.push_back( point );
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetEigenvectorAndEigenvalue( const std::vector< double >* const eigenvector, 
                                                                                  double eigenvalue ) 
{
  // Set the eigenvector
  std::vector< double > tempEigenvector;

  for ( unsigned int i=0; i<this->NumberOfSurfacePoints; i++ )
    {
    tempEigenvector.push_back( (*eigenvector)[i] );
    }
  this->Eigenvectors.push_back( tempEigenvector );

  // Set the eigenvalue
  this->Eigenvalues.push_back( eigenvalue );
  // Increment to keep track of the number of modes
  this->NumberOfModes++;
}

#endif
