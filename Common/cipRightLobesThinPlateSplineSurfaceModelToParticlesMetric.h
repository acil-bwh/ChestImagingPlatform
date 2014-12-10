/**
 *  \file cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric
 *  \ingroup common
 *  \brief This class 
 *
 */

#ifndef __cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric_h
#define __cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric_h

#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"

class cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric: public cipThinPlateSplineSurfaceModelToParticlesMetric
{
public:
  cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric();
  ~cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric();

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. */
  double GetValue( const std::vector< double >* const ); 

  const cipThinPlateSplineSurface& GetRightHorizontalThinPlateSplineSurface()
    {
      return RightHorizontalNewtonOptimizer.GetMetric().GetThinPlateSplineSurface();
    }

  const cipThinPlateSplineSurface& GetRightObliqueThinPlateSplineSurface()
    {
      return RightObliqueNewtonOptimizer.GetMetric().GetThinPlateSplineSurface();
    }

private:
  double GetFissureTermValue();
  double GetVesselTermValue();
  double GetAirwayTermValue();

  std::vector< cip::PointType > RightObliqueSurfacePoints;
  std::vector< cip::PointType > RightHorizontalSurfacePoints;

  cipNewtonOptimizer< 2 >  RightObliqueNewtonOptimizer;
  /* cipThinPlateSplineSurface                    RightObliqueThinPlateSplineSurface; */
  /* cipParticleToThinPlateSplineSurfaceMetric    RightObliqueParticleToTPSMetric; */

  cipNewtonOptimizer< 2 >  RightHorizontalNewtonOptimizer;
  /* cipThinPlateSplineSurface                    RightHorizontalThinPlateSplineSurface; */
  /* cipParticleToThinPlateSplineSurfaceMetric    RightHorizontalParticleToTPSMetric; */
};


#endif
