/**
 *
 *  $Date: 2012-08-28 17:54:34 -0400 (Tue, 28 Aug 2012) $
 *  $Revision: 213 $
 *  $Author: jross $
 *
 */

#include "cipVesselParticleConnectedComponentFilter.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkSmartPointer.h"
#include <cfloat>
#include "itkImageFileWriter.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"


cipVesselParticleConnectedComponentFilter::cipVesselParticleConnectedComponentFilter()
{
  this->ScaleRatioThreshold       = 1.0;
  this->MaxAllowableScale         = DBL_MAX;
  this->MinAllowableScale         = 0;
}


vtkPolyData* cipVesselParticleConnectedComponentFilter::GetOutput()
{
  return this->OutputPolyData;
}

void cipVesselParticleConnectedComponentFilter::SetScaleRatioThreshold( double threshold )
{
  this->ScaleRatioThreshold = threshold;
}

void cipVesselParticleConnectedComponentFilter::SetMaximumAllowableScale( double maxScale )
{
  this->MaxAllowableScale = maxScale;
}

void cipVesselParticleConnectedComponentFilter::SetMinimumAllowableScale( double minScale )
{
  this->MinAllowableScale = minScale;
}

double cipVesselParticleConnectedComponentFilter::GetScaleRatioThreshold()
{
  return this->ScaleRatioThreshold;
}


bool cipVesselParticleConnectedComponentFilter::EvaluateParticleConnectedness( unsigned int particleIndex1, unsigned int particleIndex2 )
{
  // Evaluate whether or not the two particls are sufficiently of the
  // same scale
  double scale1 = this->InputPolyData->GetPointData()->GetArray( "scale" )->GetTuple( particleIndex1 )[0];
  double scale2 = this->InputPolyData->GetPointData()->GetArray( "scale" )->GetTuple( particleIndex2 )[0];

  if ( scale1 < this->MinAllowableScale ||  scale2 < this->MinAllowableScale)
    {
      return false;
    }
  
  if ( scale1 > this->MaxAllowableScale || scale2 > this->MaxAllowableScale )
    {
      return false;
    }

  double maxScale;  (scale1>scale2) ? (maxScale = scale1) : (maxScale = scale2);

  if ( std::abs(scale1 - scale2)/maxScale > this->ScaleRatioThreshold )
    {
    return false;
    }

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
    particle1Hevec2[0] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex1 )[0];
    particle1Hevec2[1] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex1 )[1];
    particle1Hevec2[2] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex1 )[2];

  double particle2Hevec2[3];
    particle2Hevec2[0] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex2 )[0];
    particle2Hevec2[1] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex2 )[1];
    particle2Hevec2[2] = this->InputPolyData->GetPointData()->GetArray( "hevec0" )->GetTuple( particleIndex2 )[2];

  if ( this->GetVectorMagnitude( connectingVec ) > this->ParticleDistanceThreshold )
    {
    return false;
    }

  double theta1 = this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double theta2 = this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

  if ( theta1 > this->ParticleAngleThreshold || theta2 > this->ParticleAngleThreshold )
    {
    return false;
    } 

  return true;
}


