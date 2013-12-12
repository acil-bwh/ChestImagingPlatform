/**
 *  \file cipThinPlateSplineSurfaceModelToParticlesMetric
 *  \ingroup common
 *  \brief This class implements an objective function that measures how well
 *  a given thin plate spline (TPS) surface model fits a given set of
 *  particles data. The intuition behind the metric is that particles
 *  that are close to the current surface and oriented parallel to the
 *  direction of the surface normal (at the nearest point) contribute
 *  strongly to the objective function value. On the other hand,
 *  particles that are far from the surface and/or oriented orthogonal
 *  to the surface normal do not contribute to the objective function
 *  value.
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */

#ifndef __cipThinPlateSplineSurfaceModelToParticlesMetric_h
#define __cipThinPlateSplineSurfaceModelToParticlesMetric_h

#include "vtkPolyData.h"
#include "cipThinPlateSplineSurface.h"
#include "cipNewtonOptimizer.h"
#include "cipParticleToThinPlateSplineSurfaceMetric.h"

class cipThinPlateSplineSurfaceModelToParticlesMetric
{
public:
  cipThinPlateSplineSurfaceModelToParticlesMetric();
  ~cipThinPlateSplineSurfaceModelToParticlesMetric();

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. */
  double GetValue( const std::vector< double >* const ) const; 

  /** Set the particles data to be used during the optimization */     
  void SetParticles( vtkPolyData* const );

  void SetParticleWeights( std::vector< double >* );

  /** The mean surface points are the physical x, y, and z-coordinates
   *  of a collection of points that define the mean surface in our model */
  void SetMeanSurfacePoints( const std::vector< double* >* const );

  /** Set the surface model eigenvectors and eigenvalues with this
   *  method. Note that the eigenvectors are normalized, and they
   *  represent the z-coordinates of the control points. There should
   *  be the same number of entries in each eigenvector as there are
   *  entries for the collection of mean surface points. This method
   *  should be called multiple times (once for each eigenvector,
   *  eigenvalue pair) to establish the full shape model */
  void SetEigenvectorAndEigenvalue( const std::vector< double >* const, double );

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
  vtkPolyData* Particles;

  cipNewtonOptimizer< 2 >*                     NewtonOptimizer;
  cipThinPlateSplineSurface*                   ThinPlateSplineSurface;
  cipParticleToThinPlateSplineSurfaceMetric*   ParticleToTPSMetric;

  std::vector< double >                  ParticleWeights;
  std::vector< double* >                 SurfacePoints;
  std::vector< std::vector< double > >   Eigenvectors;
  std::vector< double >                  Eigenvalues;
  std::vector< double* >                 MeanPoints;

  double SigmaDistance;
  double SigmaTheta;

  unsigned int NumberOfModes;
  unsigned int NumberOfSurfacePoints;
  unsigned int NumberOfParticles;
};


#endif
