/**
 *  \file cipParticleToThinPlateSplineSurfaceMetric
 *  \ingroup common
 *  \brief  This class implements and objective function that is used to
 *  determine the closest point on a thin plate spline (TPS) surface
 *  for a given particle location.
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */

#ifndef __cipParticleToThinPlateSplineSurfaceMetric_h
#define __cipParticleToThinPlateSplineSurfaceMetric_h

#include "cipThinPlateSplineSurface.h"
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>


class cipParticleToThinPlateSplineSurfaceMetric
{
public:
  cipParticleToThinPlateSplineSurfaceMetric() {};
  ~cipParticleToThinPlateSplineSurfaceMetric() {};

  typedef vnl_vector< double >   VectorType;
  typedef vnl_vector< double >   PointType;
  typedef vnl_matrix< double >   MatrixType;

  /** This method returns the value of the cost function corresponding
    * to the specified parameters.    */ 
  double GetValue( PointType* ) const;

  /** This method returns the value and gradient of the cost function
    * corresponding to the specified parameters.    */ 
  double GetValueAndGradient( PointType*, VectorType* ) const;

  /** This method returns the value, gradient, and Hessian of the cost
    * function corresponding to the specified parameters.    */ 
  double GetValueGradientAndHessian( PointType*, VectorType*, MatrixType* ) const;

  /** Set the x, y, and z coordinates of the particle */
  void SetParticle( cip::PointType );

  void SetThinPlateSplineSurface( const cipThinPlateSplineSurface& );

  /** Expose the TPS surface so that it can be modified */
  cipThinPlateSplineSurface& GetThinPlateSplineSurface()
  {
    return ThinPlateSplineSurface;
  }

private:
  cipThinPlateSplineSurface ThinPlateSplineSurface;

  double GetVectorMagnitude( const double[3] ) const;
  double GetAngleBetweenVectors( const double[3], const double[3] ) const;

  double ParticlePosition[3];
};


#endif
