/**
 *  \class cipVesselParticleConnectedComponentFilter
 *  \ingroup common
 *  \brief  This filter is designed to filter noisy particles. It is
 *  based on connected component concepts. See inherited class header
 *  for more details.
 *
 *  For the vessel particles, connection is assessed by the degree to
 *  which nearby particles define a local cylinder. The relative scale
 *  between nearby particles is also assessed. Provided that two
 *  nearby particles are approximately the same scale and sufficiently
 *  define a local cylinder, they will be grouped in the same
 *  connected component. Only connected components having sufficiently
 *  large cardinality will appear in the output.
 *
 *  $Date: 2012-08-28 17:54:34 -0400 (Tue, 28 Aug 2012) $
 *  $Revision: 213 $
 *  $Author: jross $
 *
 */

#ifndef __cipVesselParticleConnectedComponentFilter_h
#define __cipVesselParticleConnectedComponentFilter_h


#include "cipParticleConnectedComponentFilter.h"


class cipVesselParticleConnectedComponentFilter: public cipParticleConnectedComponentFilter
{
public:

  cipVesselParticleConnectedComponentFilter();
  ~cipVesselParticleConnectedComponentFilter(){};

  /** Value to determine how close two particles must be in scale to
      be considered connected. The value should be in the interval
      [0,1]. The closer the value is to 1, the more permissive the
      filter is to large differences in scale between adjacent
      particles. */
  void   SetScaleRatioThreshold( double );
  double GetScaleRatioThreshold();

  /** When considering the connectedness between two particles, if either
   *  of them has a scale greater than the max allowable scale, no connection
   *  will be formed */
  void SetMaximumAllowableScale( double );

  /** When considering the connectedness between two particles, if either
   *  of them has a scale smaller than the min allowable scale, no connection
   *  will be formed */
  void SetMinimumAllowableScale( double );
  
  vtkPolyData* GetOutput();

private:
  bool   EvaluateParticleConnectedness( unsigned int, unsigned int );

  double ScaleRatioThreshold;
  double MaxAllowableScale;
  double MinAllowableScale;
};

#endif
