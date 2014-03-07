/**
 *  \file cipThinPlateSplineSurfaceModelToParticlesMetric
 *  \ingroup common
 *  \brief This class 
 *
 */

#ifndef __cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric_h
#define __cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric_h

#include "vtkPolyData.h"
#include "cipThinPlateSplineSurface.h"
#include "cipNewtonOptimizer.h"
#include "cipParticleToThinPlateSplineSurfaceMetric.h"

class cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric: public cipThinPlateSplineSurfaceModelToParticlesMetric
{
public:
  cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric();
  ~cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric();

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. */
  double GetValue( const std::vector< double >* const ) const; 

  /** The sigma distance parameter value controls the effect of
   *  particle-surface distance during the objective function value
   *  computation. */
  void SetSigmaDistance( double sigDist )
    {
      SigmaDistance = sigDist;
    }

  /** The sigma theta parameter value controls the effect of
   *  particle-surface orientation difference during the objective
   *  function value computation. */
  void SetSigmaTheta( double sigTheta )
    {
      SigmaTheta = sigTheta;
    }

  cipThinPlateSplineSurface* GetThinPlateSplineSurface()
    {
      return ThinPlateSplineSurface;
    }

private:
  double GetFissureTermValue() = 0;
  double GetAirwayTermValue()  = 0;
  double GetVesselTermValue()  = 0;

  cipNewtonOptimizer< 2 >*                     RightObliqueNewtonOptimizer;
  cipThinPlateSplineSurface*                   RightObliqueThinPlateSplineSurface;
  cipParticleToThinPlateSplineSurfaceMetric*   RightObliqueParticleToTPSMetric;

  cipNewtonOptimizer< 2 >*                     RightHorizontalNewtonOptimizer;
  cipThinPlateSplineSurface*                   RightHorizontalThinPlateSplineSurface;
  cipParticleToThinPlateSplineSurfaceMetric*   RightHorizontalParticleToTPSMetric;

  double SigmaDistance;
  double SigmaTheta;
};


#endif
