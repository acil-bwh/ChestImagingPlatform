/**
 *  \class cipAirwayParticleConnectedComponentFilter
 *  \ingroup common
 *  \brief  This filter is designed to filter noisy particles. It is
 *  based on connected component concepts. See inherited class header
 *  for more details.
 *
 *  For the airway particles, connection is assessed by the degree to
 *  which nearby particles define a local cylinder. The relative scale
 *  between nearby particles is also assessed. Provided that two
 *  nearby particles are approximately the same scale and sufficiently
 *  define a local cylinder, they will be grouped in the same
 *  connected component. Only connected components having sufficiently
 *  large cardinality will appear in the output.
 *
 *  $Date: 2012-09-17 18:32:23 -0400 (Mon, 17 Sep 2012) $
 *  $Revision: 268 $
 *  $Author: jross $
 *
 */

#ifndef __cipAirwayParticleConnectedComponentFilter_h
#define __cipAirwayParticleConnectedComponentFilter_h


#include "cipParticleConnectedComponentFilter.h"


class cipAirwayParticleConnectedComponentFilter: public cipParticleConnectedComponentFilter
{
public:

  cipAirwayParticleConnectedComponentFilter();
  ~cipAirwayParticleConnectedComponentFilter(){};

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
