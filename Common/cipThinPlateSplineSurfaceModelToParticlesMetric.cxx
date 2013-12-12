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
  // Particles are represented by VTK polydata
  this->Particles = vtkPolyData::New();

  // The 'cipThinPlateSplineSurface' class wraps functionality for
  // constructing and accessing data for a TPS interpolating surface
  // given a set of surface points 
  this->ThinPlateSplineSurface  = new cipThinPlateSplineSurface();

  this->ParticleToTPSMetric     = new cipParticleToThinPlateSplineSurfaceMetric();
  this->ParticleToTPSMetric->SetThinPlateSplineSurface( this->ThinPlateSplineSurface );

  this->NewtonOptimizer         = new cipNewtonOptimizer< 2 >();
  this->NewtonOptimizer->SetMetric( this->ParticleToTPSMetric );

  this->NumberOfModes     = 0; 
  this->NumberOfParticles = 0;
}


cipThinPlateSplineSurfaceModelToParticlesMetric
::~cipThinPlateSplineSurfaceModelToParticlesMetric()
{
}


void cipThinPlateSplineSurfaceModelToParticlesMetric::SetParticles( vtkPolyData* const particles )
{
  this->Particles = particles;
  this->NumberOfParticles = this->Particles->GetNumberOfPoints();

  // If no particle weights have already been specified, set each
  // particle to have equal, unity weight
  if ( this->ParticleWeights.size() == 0 )
    {
    for ( unsigned int i=0; i<this->NumberOfParticles; i++ )
      {
      this->ParticleWeights.push_back( 1.0 );
      }
    }
}


void cipThinPlateSplineSurfaceModelToParticlesMetric::SetParticleWeights( std::vector< double >* weights )
{
  //
  // Clear any existing particle weights first
  //
  this->ParticleWeights.clear();

  for ( unsigned int i=0; i<weights->size(); i++ )
    {
    this->ParticleWeights.push_back( (*weights)[i] );
    }
}


//
// Set the surface points of the mean surface model. These points --
// in conjunction with the various model eigenvectors and eigenvalues
// -- will be used to construct the TPS surfaces
//
void cipThinPlateSplineSurfaceModelToParticlesMetric::SetMeanSurfacePoints( const std::vector< double* >* const meanPoints ) 
{
  this->NumberOfSurfacePoints = meanPoints->size();

  for ( unsigned int i=0; i<this->NumberOfSurfacePoints; i++ )
    {
    double* meanPoint = new double[3];
      meanPoint[0] = (*meanPoints)[i][0];
      meanPoint[1] = (*meanPoints)[i][1];
      meanPoint[2] = (*meanPoints)[i][2];

    this->MeanPoints.push_back( meanPoint );
    
    //
    // We will also initialize 'SurfacePoints' here. Calls to the
    // 'GetValue' method will alter the z-values of 'SurfacePoints',
    // but the x and y locations will remain the same no matter what. 
    //
    double* surfacePoint = new double[3];
      surfacePoint[0] = (*meanPoints)[i][0];
      surfacePoint[1] = (*meanPoints)[i][1];
      surfacePoint[2] = (*meanPoints)[i][2];

    this->SurfacePoints.push_back( surfacePoint );
    }
}

void cipThinPlateSplineSurfaceModelToParticlesMetric::SetEigenvectorAndEigenvalue( const std::vector< double >* const eigenvector, 
                                                                                  double eigenvalue ) 
{
  //
  // Set the eigenvector
  //
  std::vector< double > tempEigenvector;

  for ( unsigned int i=0; i<this->NumberOfSurfacePoints; i++ )
    {
    tempEigenvector.push_back( (*eigenvector)[i] );
    }

  this->Eigenvectors.push_back( tempEigenvector );

  //
  // Set the eigenvalue
  //
  this->Eigenvalues.push_back( eigenvalue );

  //
  // Increment to keep track of the number of modes
  //
  this->NumberOfModes++;
}


//
// Note that 'params' must have the same number of entries as the
// number of PCA modes in our model. Also note that
// 'SetMeanSurfacePoints' must be called prior to the 'GetValue'
// call. 
//
double cipThinPlateSplineSurfaceModelToParticlesMetric::GetValue( const std::vector< double >* const params ) const
{
  //
  // First we must construct the TPS surface given the param values
  //
  for ( unsigned int p=0; p<this->NumberOfSurfacePoints; p++ )
    {    
    this->SurfacePoints[p][2] = this->MeanPoints[p][2];

    for ( unsigned int m=0; m<this->NumberOfModes; m++ )      
      {
      //
      // Note that we need only adjust the z-coordinate, as the domain
      // locations of the points (the x and y coordinates) remain
      // fixed 
      //
      this->SurfacePoints[p][2] += (*params)[m]*vcl_sqrt(this->Eigenvalues[m])*this->Eigenvectors[m][p];
      }
    }

  //
  // Now that we have our surface points, we can construct the TPS
  // surface 
  //
  this->ThinPlateSplineSurface->SetSurfacePoints( &this->SurfacePoints );

  //
  // We now have our TPS surface. At this point we need to loop over
  // all the particles and tally up our metric
  //
  double* position    = new double[3];
  double* normal      = new double[3];
  double* orientation = new double[3];

  cipNewtonOptimizer< 2 >::PointType* domainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* optimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  double value = 0.0;
  
  //
  // We compute the following coefficient here. It is used to
  // normalize the value of the particle's contribution to the value
  //
  double coefficient = 1.0/(2.0*vnl_math::pi*this->SigmaTheta*this->SigmaDistance);

  for ( unsigned int i=0; i<this->NumberOfParticles; i++ )
    {
    position[0] = this->Particles->GetPoint(i)[0];
    position[1] = this->Particles->GetPoint(i)[1];
    position[2] = this->Particles->GetPoint(i)[2];

    orientation[0] = this->Particles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[0];
    orientation[1] = this->Particles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[1];
    orientation[2] = this->Particles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[2];

    //
    // Determine the domain location for which the particle is closest
    // to the TPS surface
    //
    this->ParticleToTPSMetric->SetParticle( position );

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
    this->NewtonOptimizer->SetInitialParameters( domainParams );
    this->NewtonOptimizer->Update();
    this->NewtonOptimizer->GetOptimalParameters( optimalParams );

    //
    // Get the distance between the particle and the TPS surface. This
    // is just the square root of the objective function value
    // optimized by the Newton method.
    //
    double distance = vcl_sqrt( this->NewtonOptimizer->GetOptimalValue() );

    //
    // Get the TPS surface normal at the domain location.
    //    
    this->ThinPlateSplineSurface->GetSurfaceNormal( (*optimalParams)[0], (*optimalParams)[1], normal );
    double theta = cip::GetAngleBetweenVectors( normal, orientation, true );

    //
    // Now that we have the surface normal and distance, we can
    // compute this particle's contribution to the overall objective
    // function value
    //    
    value -= this->ParticleWeights[i]*coefficient*std::exp( -0.5*std::pow(distance/this->SigmaDistance,2) )*
      std::exp( -0.5*std::pow(theta/this->SigmaTheta,2) );
    }

  delete position;
  delete normal;
  delete orientation;
  
  return value;
}

#endif
