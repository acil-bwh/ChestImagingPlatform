/**
 *  \file cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric *  \ingroup common
 *  \brief This class 
 *
 */

#ifndef __cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric_h
#define __cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric_h

#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"

class cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric: public cipThinPlateSplineSurfaceModelToParticlesMetric
{
public:
  cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric();
  ~cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric();

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. */
  double GetValue( const std::vector< double >* const ); 

  cipThinPlateSplineSurface* GetLeftObliqueThinPlateSplineSurface()
    {
      return LeftObliqueThinPlateSplineSurface;
    }

private:
  double GetFissureTermValue();
  double GetVesselTermValue();
  double GetAirwayTermValue();

  std::vector< double* > LeftObliqueSurfacePoints;

  cipNewtonOptimizer< 2 >*                     LeftObliqueNewtonOptimizer;
  cipThinPlateSplineSurface*                   LeftObliqueThinPlateSplineSurface;
  cipParticleToThinPlateSplineSurfaceMetric*   LeftObliqueParticleToTPSMetric;
};


#endif
