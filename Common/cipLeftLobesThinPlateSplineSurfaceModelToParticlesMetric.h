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

  const cipThinPlateSplineSurface& GetLeftObliqueThinPlateSplineSurface()
    {
      return LeftObliqueNewtonOptimizer.GetMetric().GetThinPlateSplineSurface();
    }

private:
  double GetFissureTermValue();
  double GetVesselTermValue();
  double GetAirwayTermValue();

  std::vector< cip::PointType > LeftObliqueSurfacePoints;

  cipNewtonOptimizer< 2 >  LeftObliqueNewtonOptimizer;
};


#endif
