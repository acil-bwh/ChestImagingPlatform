#include "itkImageFileWriter.h"
#include "cipFissureParticleConnectedComponentFilter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include <cfloat>
#include "itkImageFileWriter.h"


cipFissureParticleConnectedComponentFilter::cipFissureParticleConnectedComponentFilter()
{
  this->ParticleDistanceThreshold = 3.0;
  this->ParticleAngleThreshold    = 70.0;
}


bool cipFissureParticleConnectedComponentFilter::EvaluateParticleConnectedness( unsigned int particleIndex1, unsigned int particleIndex2 )
{
  //
  // Determine the vector connecting the two particles
  //
  double point1[3];
    point1[0] = this->InputPolyData->GetPoint( particleIndex1 )[0];
    point1[1] = this->InputPolyData->GetPoint( particleIndex1 )[1];
    point1[2] = this->InputPolyData->GetPoint( particleIndex1 )[2];

  double point2[3];
    point2[0] = this->InputPolyData->GetPoint( particleIndex2 )[0];
    point2[1] = this->InputPolyData->GetPoint( particleIndex2 )[1];
    point2[2] = this->InputPolyData->GetPoint( particleIndex2 )[2];

  double connectingVec[3];
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double particle1Hevec2[3];
    particle1Hevec2[0] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[0];
    particle1Hevec2[1] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[1];
    particle1Hevec2[2] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[2];

  double particle2Hevec2[3];
    particle2Hevec2[0] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[0];
    particle2Hevec2[1] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[1];
    particle2Hevec2[2] = this->InputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[2];
	
  if ( this->GetVectorMagnitude( connectingVec ) > this->ParticleDistanceThreshold )
    {
      return false;
    }

  double theta1 = this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double theta2 = this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

  if ( theta1 < this->ParticleAngleThreshold || theta2 < this->ParticleAngleThreshold )
    {
    return false;
    } 

  return true;
}
