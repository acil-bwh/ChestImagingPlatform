/**
 *  \class cipFissureParticleConnectedComponentFilter
 *  \ingroup common
 *  \brief This filter can be used to isolate large connected components
 *  within fissure particles data sets. It uses connected components
 *  concepts to accomplish this: particles lying close together and
 *  representing the same locally planar surface are grouped together,
 *  and then only the largest groups survive the filtering process.
 * 
 *  The input to the filter is a VTK poly data file. It is assumed that 
 *  there is a field data array entry called 'hevec2' representing the
 *  Hessian eigenvector pointing in the direction perpendicular to the
 *  fissure plane. The output dataset has the same field data array
 *  entries as the input, but with an addional array named 
 *  'unmergedComponents' indicating the component label assigned to a
 *  specific particle.
 *
 *  $Date$
 *  $Revision $
 *  $Author$
 * 
 */

#ifndef __cipFissureParticleConnectedComponentFilter_h
#define __cipFissureParticleConnectedComponentFilter_h


#include "cipParticleConnectedComponentFilter.h"


class cipFissureParticleConnectedComponentFilter: public cipParticleConnectedComponentFilter
{
public:
  ~cipFissureParticleConnectedComponentFilter(){};
  cipFissureParticleConnectedComponentFilter();

private:
  /** Overrides the base-class implementation to indicate the specific 
      criteria needed for evaluating connectedness amongst fissure 
      particles */
  bool EvaluateParticleConnectedness( unsigned int, unsigned int );
};

#endif
