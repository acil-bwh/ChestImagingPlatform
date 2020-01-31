/**
 *
 */

#ifndef __cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric_cxx
#define __cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric_cxx

#include "cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "cipHelper.h"
#include "cipMacro.h"
#include "cipExceptionObject.h"

cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric
::cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric()
{
}


cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric
::~cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric()
{
}

// Note that 'params' must have the same number of entries as the
// number of PCA modes in our model. Also note that
// 'SetMeanSurfacePoints' must be called prior to the 'GetValue'
// call. 
double cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric::GetValue( const std::vector< double >* const params )
{
  if ( this->MeanPoints.size() == 0 )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, 
				  "cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric::GetValue( const std::vector< double >* const params )", 
				  "Mean surface points are not set" );
    }

  cipAssert ( this->MeanPoints.size() == this->NumberOfSurfacePoints );

  if ( this->LeftObliqueSurfacePoints.size() == 0 )
    {
      for ( unsigned int i=0; i<this->NumberOfSurfacePoints; i++ )
	{
	  cip::PointType tmp(3);
	    tmp[0] = this->MeanPoints[i][0];
	    tmp[1] = this->MeanPoints[i][1];
	    tmp[2] = this->MeanPoints[i][2];

	  this->LeftObliqueSurfacePoints.push_back( tmp );
	}
    }

  cipAssert ( this->LeftObliqueSurfacePoints.size() == this->NumberOfSurfacePoints );
  cipAssert ( this->Eigenvalues.size() == this->NumberOfModes );
  cipAssert ( this->Eigenvectors.size() == this->NumberOfModes );

  // First we must construct the TPS surface given the param values. 
  for ( unsigned int p=0; p<this->NumberOfSurfacePoints; p++ )
    {
      this->LeftObliqueSurfacePoints[p][2] = this->MeanPoints[p][2];

      for ( unsigned int m=0; m<this->NumberOfModes; m++ )      
	{
	  // Note that we need only adjust the z-coordinate, as the domain
	  // locations of the points (the x and y coordinates) remain
	  // fixed 
	  this->LeftObliqueSurfacePoints[p][2] += 
	    (*params)[m]*std::sqrt(this->Eigenvalues[m])*this->Eigenvectors[m][p];
	}
    }

  // Now that we have our surface points, we can construct the TPS
  // surfaces corresponding to the left oblique boundaries
  this->LeftObliqueNewtonOptimizer.GetMetric().GetThinPlateSplineSurface().
    SetSurfacePoints( this->LeftObliqueSurfacePoints );

  double regularizer = 0;
  for ( unsigned int m=0; m<this->NumberOfModes; m++ )      
    {
      regularizer += std::pow((*params)[m], 2);
    }

  double fissureTermValue = this->GetFissureTermValue();
  double vesselTermValue = this->GetVesselTermValue();

  // Note that we multiply the vessel term weight by 500.0. This is because at convergence,
  // the fissue metric value per fissure particle is roughtly 500 times that of the 
  // vessel metric value per vessel particle. We want these two terms to be roughly equal
  // so that the user-defined term weights are intuitive to use.
  double value = this->FissureTermWeight*fissureTermValue + 
    500.0*this->VesselTermWeight*vesselTermValue +
    this->RegularizationWeight*regularizer;

  return value;
}

double cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric::GetFissureTermValue()
{
  double fissureTermValue = 0.0;

  cip::PointType position(3);
  cip::VectorType loNormal(3);
  cip::VectorType orientation(3);

  cipNewtonOptimizer< 2 >::PointType* loDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* loOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  for ( unsigned int i=0; i<this->NumberOfFissureParticles; i++ )
    {
    position[0] = this->FissureParticles->GetPoint(i)[0];
    position[1] = this->FissureParticles->GetPoint(i)[1];
    position[2] = this->FissureParticles->GetPoint(i)[2];

    orientation[0] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[0];
    orientation[1] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[1];
    orientation[2] = this->FissureParticles->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[2];

    // Determine the domain locations for which the particle is closest
    // to the left oblique
    this->LeftObliqueNewtonOptimizer.GetMetric().SetParticle( position );

    // The particle's x, and y location are a good place to initialize
    // the search for the domain locations that result in the smallest
    // distance between the particle and the TPS surfaces
    (*loDomainParams)[0] = position[0]; 
    (*loDomainParams)[1] = position[1]; 

    // Perform Newton line search to determine the closest point on
    // the current TPS surfaces
    this->LeftObliqueNewtonOptimizer.SetInitialParameters( loDomainParams );
    this->LeftObliqueNewtonOptimizer.Update();
    this->LeftObliqueNewtonOptimizer.GetOptimalParameters( loOptimalParams );

    // Get the distances between the particle and the TPS surfaces. This
    // is just the square root of the objective function value
    // optimized by the Newton method.
    double loDistance = std::sqrt( this->LeftObliqueNewtonOptimizer.GetOptimalValue() );

    // Get the TPS surface normals at the domain locations.
    this->LeftObliqueNewtonOptimizer.GetMetric().GetThinPlateSplineSurface().
      GetSurfaceNormal( (*loOptimalParams)[0], (*loOptimalParams)[1], loNormal );
    double loTheta = cip::GetAngleBetweenVectors( loNormal, orientation, true );

    // Now that we have the surface normals and distances, we can compute this 
    // particle's contribution to the overall objective function value. 
    fissureTermValue -= this->FissureParticleWeights[i]*std::exp( -loDistance/this->FissureSigmaDistance )*
      std::exp( -loTheta/this->FissureSigmaTheta );
    }

  return fissureTermValue;
}

double cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric::GetVesselTermValue()
{
  double vesselTermValue = 0.0;

  cip::PointType position(3);
  cip::VectorType loNormal(3);
  cip::VectorType orientation(3);

  cipNewtonOptimizer< 2 >::PointType* loDomainParams  = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );
  cipNewtonOptimizer< 2 >::PointType* loOptimalParams = new cipNewtonOptimizer< 2 >::PointType( 2, 2 );

  for ( unsigned int i=0; i<this->NumberOfVesselParticles; i++ )
    {
    position[0] = this->VesselParticles->GetPoint(i)[0];
    position[1] = this->VesselParticles->GetPoint(i)[1];
    position[2] = this->VesselParticles->GetPoint(i)[2];

    orientation[0] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[0];
    orientation[1] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[1];
    orientation[2] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple(i)[2];

    // Determine the domain locations for which the particle is closest
    // to the left oblique TPS surfaces
    this->LeftObliqueNewtonOptimizer.GetMetric().SetParticle( position );

    // The particle's x, and y location are a good place to initialize
    // the search for the domain locations that result in the smallest
    // distance between the particle and the TPS surfaces
    (*loDomainParams)[0] = position[0]; 
    (*loDomainParams)[1] = position[1]; 

    // Perform Newton line search to determine the closest point on
    // the current TPS surfaces
    this->LeftObliqueNewtonOptimizer.SetInitialParameters( loDomainParams );
    this->LeftObliqueNewtonOptimizer.Update();
    this->LeftObliqueNewtonOptimizer.GetOptimalParameters( loOptimalParams );

    // Get the distances between the particle and the TPS surfaces. This
    // is just the square root of the objective function value
    // optimized by the Newton method.
    double loDistance = std::sqrt( this->LeftObliqueNewtonOptimizer.GetOptimalValue() );

    // Get the TPS surface normals at the domain locations.
    this->LeftObliqueNewtonOptimizer.GetMetric().GetThinPlateSplineSurface().
      GetSurfaceNormal( (*loOptimalParams)[0], (*loOptimalParams)[1], loNormal );

    double loTheta = cip::GetAngleBetweenVectors( loNormal, orientation, true );

    // Now that we have the surface normals and distances, we can compute this 
    // particle's contribution to the overall objective function value. 
    vesselTermValue += this->VesselParticleWeights[i]*std::exp( -loDistance/this->VesselSigmaDistance )*
      std::exp( -loTheta/this->VesselSigmaTheta );
    }

  return vesselTermValue;
}

double cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric::GetAirwayTermValue()
{
  return 0;
}

#endif
