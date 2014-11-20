/**
 *  \file cipThinPlateSplineSurfaceModelToParticlesMetric
 *  \ingroup common
 *  \brief This is an abstract base class that abstracts away some
 *  core functionality needed by classes that compute metric values
 *  between TPS surface(s) and sets of particles.
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
  virtual double GetValue( const std::vector< double >* const ) = 0; 

  /** Set fissure particles data that may be used during the optimization */
  void SetFissureParticles( vtkPolyData* const );

  /** Set airway particles data that may be used during the optimization */
  void SetAirwayParticles( vtkPolyData* const );

  /** Set vessel particles data that may be used during the optimization */
  void SetVesselParticles( vtkPolyData* const );

  /** Set weights for each of the fissure particles */
  void SetFissureParticleWeights( std::vector< double >* );

  /** Set weights for each of the airway particles */
  void SetAirwayParticleWeights( std::vector< double >* );

  /** Set weights for each of the vessel particles */
  void SetVesselParticleWeights( std::vector< double >* );

  /** The mean surface points are the physical x, y, and z-coordinates
   *  of a collection of points that define the mean surface in our model */
  void SetMeanSurfacePoints( const std::vector< cip::PointType >& );

  /** Set the surface model eigenvectors and eigenvalues with this
   *  method. Note that the eigenvectors are normalized, and they
   *  represent the z-coordinates of the control points. There should
   *  be the same number of entries in each eigenvector as there are
   *  entries for the collection of mean surface points. This method
   *  should be called multiple times (once for each eigenvector,
   *  eigenvalue pair) to establish the full shape model */
  void SetEigenvectorAndEigenvalue( const std::vector< double >* const, double );

  /** The fissure sigma distance parameter value controls the effect of
   *  fissure particle-surface distance during the objective function value
   *  computation. */
  void SetFissureSigmaDistance( double sigDist )
    {
      FissureSigmaDistance = sigDist;
    }

  /** The sigma theta parameter value controls the effect of fissure
   *  particle-surface orientation difference during the objective
   *  function value computation. */
  void SetFissureSigmaTheta( double sigTheta )
    {
      FissureSigmaTheta = sigTheta;
    }

  /** The vessel sigma distance parameter value controls the effect of
   *  vessel particle-surface distance during the objective function value
   *  computation. */
  void SetVesselSigmaDistance( double sigDist )
    {
      VesselSigmaDistance = sigDist;
    }

  /** The sigma theta parameter value controls the effect of vessel
   *  particle-surface orientation difference during the objective
   *  function value computation. */
  void SetVesselSigmaTheta( double sigTheta )
    {
      VesselSigmaTheta = sigTheta;
    }

  /** The airway sigma distance parameter value controls the effect of
   *  airway particle-surface distance during the objective function value
   *  computation. */
  void SetAirwaySigmaDistance( double sigDist )
    {
      AirwaySigmaDistance = sigDist;
    }

  /** The sigma theta parameter value controls the effect of airway
   *  particle-surface orientation difference during the objective
   *  function value computation. */
  void SetAirwaySigmaTheta( double sigTheta )
    {
      AirwaySigmaTheta = sigTheta;
    }

  /** In general the metric will be composed of airway, fissure, and vessel
      terms. Optionally set the weight of the fissure term with this 
      function. */
  void SetFissureTermWeight( double weight )
    {
      FissureTermWeight = weight;
    }

  /** In general the metric will be composed of airway, fissure, and vessel
      terms. Optionally set the weight of the vessel term with this 
      function. */
  void SetVesselTermWeight( double weight )
    {
      VesselTermWeight = weight;
    }

  /** In general the metric will be composed of airway, fissure, and vessel
      terms. Optionally set the weight of the airway term with this 
      function. */
  void SetAirwayTermWeight( double weight )
    {
      AirwayTermWeight = weight;
    }

  void SetRegularizationWeight( double weight )
  {
    RegularizationWeight = weight;
  }

  double GetRegularizationWeight()
  {
    return RegularizationWeight;
  }

  const std::vector< cip::PointType >& GetMeanSurfacePoints() const
    {
      return MeanPoints;
    }

  const std::vector< cip::PointType >& GetSurfacePoints() const
    {
      return SurfacePoints;
    }

protected:
  virtual double GetFissureTermValue() = 0;
  virtual double GetAirwayTermValue()  = 0;
  virtual double GetVesselTermValue()  = 0;

  vtkPolyData* FissureParticles;
  vtkPolyData* AirwayParticles;
  vtkPolyData* VesselParticles;

  double FissureTermWeight;
  double VesselTermWeight;
  double AirwayTermWeight;

  std::vector< double >                 FissureParticleWeights;
  std::vector< double >                 AirwayParticleWeights;
  std::vector< double >                 VesselParticleWeights;
  std::vector< cip::PointType >         SurfacePoints;
  std::vector< std::vector< double > >  Eigenvectors;
  std::vector< double >                 Eigenvalues;
  std::vector< cip::PointType >         MeanPoints;

  double FissureSigmaDistance;
  double FissureSigmaTheta;

  double VesselSigmaDistance;
  double VesselSigmaTheta;

  double AirwaySigmaDistance;
  double AirwaySigmaTheta;

  double RegularizationWeight;

  unsigned int NumberOfModes;
  unsigned int NumberOfSurfacePoints;
  unsigned int NumberOfFissureParticles;
  unsigned int NumberOfAirwayParticles;
  unsigned int NumberOfVesselParticles;
};


#endif
