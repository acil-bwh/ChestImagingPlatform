/**
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
  // The 'cipThinPlateSplineSurface' class wraps functionality for
  // constructing and accessing data for a TPS interpolating surface
  // given a set of surface points 
  this->RightObliqueThinPlateSplineSurface    = new cipThinPlateSplineSurface();
  this->RightHorizontalThinPlateSplineSurface = new cipThinPlateSplineSurface();

  this->RightObliqueParticleToTPSMetric = new cipParticleToThinPlateSplineSurfaceMetric();
  this->RightObliqueParticleToTPSMetric->SetThinPlateSplineSurface( this->RightObliqueThinPlateSplineSurface );

  this->RightHorizontalParticleToTPSMetric = new cipParticleToThinPlateSplineSurfaceMetric();
  this->RightHorizontalParticleToTPSMetric->SetThinPlateSplineSurface( this->RightHorizontalThinPlateSplineSurface );

  this->RightObliqueNewtonOptimizer = new cipNewtonOptimizer< 2 >();
  this->RightObliqueNewtonOptimizer->SetMetric( this->RightObliqueParticleToTPSMetric );

  this->RightHorizontalNewtonOptimizer = new cipNewtonOptimizer< 2 >();
  this->RightHorizontalNewtonOptimizer->SetMetric( this->RightObliqueParticleToTPSMetric );
}


cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric
::~cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric()
{
}

// Note that 'params' must have the same number of entries as the
// number of PCA modes in our model. Also note that
// 'SetMeanSurfacePoints' must be called prior to the 'GetValue'
// call. 
double cipThinPlateSplineSurfaceModelToParticlesMetric::GetValue( const std::vector< double >* const params ) const
{
  // First we must construct the TPS surface given the param values
  for ( unsigned int p=0; p<this->NumberOfSurfacePoints; p++ )
    {
      if ( p < this->NumberOfRightObliqueSurfacePoints )
	{
	  this->RightObliqueSurfacePoints[p][2] = this->MeanPoints[p][2];

	  for ( unsigned int m=0; m<this->NumberOfModes; m++ )      
	    {
	      // Note that we need only adjust the z-coordinate, as the domain
	      // locations of the points (the x and y coordinates) remain
	      // fixed 
	      this->RightObliqueSurfacePoints[p][2] += 
		(*params)[m]*vcl_sqrt(this->Eigenvalues[m])*this->Eigenvectors[m][p];
	    }
	}
      else
	{
	  unsigned int index = p - this->NumberOfRightObliqueSurfacePoints;
	  this->RightHorizontalSurfacePoints[index][2] = this->MeanPoints[p][2];

	  for ( unsigned int m=0; m<this->NumberOfModes; m++ )      
	    {
	      // Note that we need only adjust the z-coordinate, as the domain
	      // locations of the points (the x and y coordinates) remain
	      // fixed 
	      this->RightHorizontalSurfacePoints[index][2] += 
		(*params)[m]*vcl_sqrt(this->Eigenvalues[m])*this->Eigenvectors[m][p];
	    }
	}
    }

  // Now that we have our surface points, we can construct the TPS
  // surfaces corresponding to the right horizontal and right oblique
  // boundaries
  this->RightHorizontalThinPlateSplineSurface->SetSurfacePoints( &this->RightHorizontalSurfacePoints );
  this->RightObliqueThinPlateSplineSurface->SetSurfacePoints( &this->RightObliqueSurfacePoints );

  value = this->FissureTermWeight*this->GetFissureTermValue() + this->VesselTermWeight*this->GetVesselTermValue();
  
  return value;
}

double cipThinPlateSplineSurfaceModelToParticlesMetric::GetFissureTermValue() const
{
  double fissureTermValue = 0.0;

  double* position    = new double[3];
  double* roNormal    = new double[3];
  double* rhNormal    = new double[3];
  double* orientation = new double[3];

  cipNewtonOptimizer< 2 >::PointType* roDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* roOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  cipNewtonOptimizer< 2 >::PointType* rhDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* rhOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  for ( unsigned int i=0; i<this->NumberOfFissureParticles; i++ )
    {
    position[0] = this->FissureParticles->GetPoint(i)[0];
    position[1] = this->FissureParticles->GetPoint(i)[1];
    position[2] = this->FissureParticles->GetPoint(i)[2];

    orientation[0] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[0];
    orientation[1] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[1];
    orientation[2] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[2];

    // Determine the domain locations for which the particle is closest
    // to the right oblique and right horizontal TPS surfaces
    this->RightObliqueParticleToTPSMetric->SetParticle( position );
    this->RightHorizontalParticleToTPSMetric->SetParticle( position );

    // The particle's x, and y location are a good place to initialize
    // the search for the domain locations that result in the smallest
    // distance between the particle and the TPS surfaces
    (*roDomainParams)[0] = position[0]; 
    (*roDomainParams)[1] = position[1]; 

    (*rhDomainParams)[0] = position[0]; 
    (*rhDomainParams)[1] = position[1]; 

    // Perform Newton line search to determine the closest point on
    // the current TPS surfaces
    this->RightObliqueNewtonOptimizer->SetInitialParameters( roDomainParams );
    this->RightObliqueNewtonOptimizer->Update();
    this->RightObliqueNewtonOptimizer->GetOptimalParameters( roOptimalParams );

    this->RightHorizontalNewtonOptimizer->SetInitialParameters( rhDomainParams );
    this->RightHorizontalNewtonOptimizer->Update();
    this->RightHorizontalNewtonOptimizer->GetOptimalParameters( rhOptimalParams );

    // Get the distances between the particle and the TPS surfaces. This
    // is just the square root of the objective function value
    // optimized by the Newton method.
    double roDistance = vcl_sqrt( this->RightObliqueNewtonOptimizer->GetOptimalValue() );
    double rhDistance = vcl_sqrt( this->RightHorizontalNewtonOptimizer->GetOptimalValue() );

    // Get the TPS surface normals at the domain locations.
    this->RightObliqueThinPlateSplineSurface->GetSurfaceNormal( (*roOptimalParams)[0], (*roOptimalParams)[1], roNormal );
    double roTheta = cip::GetAngleBetweenVectors( roNormal, roOrientation, true );

    this->RightHorizontalThinPlateSplineSurface->GetSurfaceNormal( (*rhOptimalParams)[0], (*rhOptimalParams)[1], rhNormal );
    double rhTheta = cip::GetAngleBetweenVectors( rhNormal, rhOrientation, true );

    // Now that we have the surface normals and distances, we can compute this 
    // particle's contribution to the overall objective function value. Note that
    // we only consider the right horizontal boundary surface provided that the
    // surface right horizontal surface point is above the right oblique surface
    // point.
    if ( this->RightHorizontalThinPlateSplineSurface->GetSurfaceHeight( (*rhOptimalParams)[0], (*rhOptimalParams)[1] ) >
	 this->RightObliqueThinPlateSplineSurface->GetSurfaceHeight( (*roOptimalParams)[0], (*roOptimalParams)[1] ) )
      {
	fissureTermValue -= this->FissureParticleWeights[i]*std::exp( -0.5*std::pow(rhDistance/this->FissureSigmaDistance,2) )*
	  std::exp( -0.5*std::pow(rhTheta/this->FissureSigmaTheta,2) );
      }
    fissureTermValue -= this->FissureParticleWeights[i]*std::exp( -0.5*std::pow(roDistance/this->FissureSigmaDistance,2) )*
      std::exp( -0.5*std::pow(roTheta/this->FissureSigmaTheta,2) );    
    }

  delete position;
  delete roNormal;
  delete rhNormal;
  delete orientation;

  return fissureTermValue;
}

double cipThinPlateSplineSurfaceModelToParticlesMetric::GetVesselTermValue() const
{
  double vesselTermValue = 0.0;

  double* position    = new double[3];
  double* roNormal    = new double[3];
  double* rhNormal    = new double[3];
  double* orientation = new double[3];

  cipNewtonOptimizer< 2 >::PointType* roDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* roOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  cipNewtonOptimizer< 2 >::PointType* rhDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* rhOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  for ( unsigned int i=0; i<this->NumberOfVesselParticles; i++ )
    {
    position[0] = this->VesselParticles->GetPoint(i)[0];
    position[1] = this->VesselParticles->GetPoint(i)[1];
    position[2] = this->VesselParticles->GetPoint(i)[2];

    orientation[0] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[0];
    orientation[1] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[1];
    orientation[2] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[2];

    // Determine the domain locations for which the particle is closest
    // to the right oblique and right horizontal TPS surfaces
    this->RightObliqueParticleToTPSMetric->SetParticle( position );
    this->RightHorizontalParticleToTPSMetric->SetParticle( position );

    // The particle's x, and y location are a good place to initialize
    // the search for the domain locations that result in the smallest
    // distance between the particle and the TPS surfaces
    (*roDomainParams)[0] = position[0]; 
    (*roDomainParams)[1] = position[1]; 

    (*rhDomainParams)[0] = position[0]; 
    (*rhDomainParams)[1] = position[1]; 

    // Perform Newton line search to determine the closest point on
    // the current TPS surfaces
    this->RightObliqueNewtonOptimizer->SetInitialParameters( roDomainParams );
    this->RightObliqueNewtonOptimizer->Update();
    this->RightObliqueNewtonOptimizer->GetOptimalParameters( roOptimalParams );

    this->RightHorizontalNewtonOptimizer->SetInitialParameters( rhDomainParams );
    this->RightHorizontalNewtonOptimizer->Update();
    this->RightHorizontalNewtonOptimizer->GetOptimalParameters( rhOptimalParams );

    // Get the distances between the particle and the TPS surfaces. This
    // is just the square root of the objective function value
    // optimized by the Newton method.
    double roDistance = vcl_sqrt( this->RightObliqueNewtonOptimizer->GetOptimalValue() );
    double rhDistance = vcl_sqrt( this->RightHorizontalNewtonOptimizer->GetOptimalValue() );

    // Get the TPS surface normals at the domain locations.
    this->RightObliqueThinPlateSplineSurface->GetSurfaceNormal( (*roOptimalParams)[0], (*roOptimalParams)[1], roNormal );
    double roTheta = cip::GetAngleBetweenVectors( roNormal, roOrientation, true );

    this->RightHorizontalThinPlateSplineSurface->GetSurfaceNormal( (*rhOptimalParams)[0], (*rhOptimalParams)[1], rhNormal );
    double rhTheta = cip::GetAngleBetweenVectors( rhNormal, rhOrientation, true );

    // Now that we have the surface normals and distances, we can compute this 
    // particle's contribution to the overall objective function value. Note that
    // we only consider the right horizontal boundary surface provided that the
    // surface right horizontal surface point is above the right oblique surface
    // point.
    if ( this->RightHorizontalThinPlateSplineSurface->GetSurfaceHeight( (*rhOptimalParams)[0], (*rhOptimalParams)[1] ) >
	 this->RightObliqueThinPlateSplineSurface->GetSurfaceHeight( (*roOptimalParams)[0], (*roOptimalParams)[1] ) )
      {
	vesselTermValue += this->VesselParticleWeights[i]*std::exp( -rhDistance/this->VesselSigmaDistance )*
	  std::exp( -rhTheta/this->VesselSigmaTheta );
      }
    
    vesselTermValue += this->VesselParticleWeights[i]*std::exp( -roDistance/this->VesselSigmaDistance )*
      std::exp( -roTheta/this->VesselSigmaTheta );
    }

  delete position;
  delete roNormal;
  delete rhNormal;
  delete orientation;

  return vesselTermValue;
}

#endif
