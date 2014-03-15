/**
 *  \file cipNelderMeadSimplexOptimizer
 *  \ingroup common
 *  \brief This class is an implementation of the Nelder-Mead simplex
 *  reflection optimizer. Calling 'Update' initializes the optimization.
 *  Note that the metric to optimize must be set with 'SetMetric',
 *  the initial parameter values must be set with
 *  'SetInitialParameters', and 'SetNumberOfIterations' must be set
 *  with the number of desired optimization iterations prior to calling
 *  'Update'. After the specified number of iterations have been
 *  executed, the user can get the optimal value and parameters using
 *  'GetOptimalValue' and 'GetOptimalParameters', respectively.
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */

#ifndef __cipNelderMeadSimplexOptimizer_h
#define __cipNelderMeadSimplexOptimizer_h

#include "cipThinPlateSplineSurface.h"
#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>

class cipNelderMeadSimplexOptimizer
{
public:
  /** This constructor expects the dimension of the objective function
   *  to be passed to it */
  cipNelderMeadSimplexOptimizer( unsigned int );
  ~cipNelderMeadSimplexOptimizer();

  /** The current implement expects a metric of type
   * 'ThinPlateSplineSurfaceModelToParticlesMetric'. This method must
   *  be called prior to 'Update' in order for the optimizer to know
   *  what to optimize. */
  void SetMetric( cipThinPlateSplineSurfaceModelToParticlesMetric* m )
    {
      Metric = m;
    };

  /** Set initial parameters to optimize */
  void SetInitialParameters( double* );

  /** Set the dimension of the objective function. Alternatively, and
   *  instantiation of this class can use the contructor that expects
   *  the dimension to be passed to it */
  void SetDimension( unsigned int dim )
    {
      Dimension = dim;
    }

  /** Tell the optimizer how many iterations to execute */
  void SetNumberOfIterations( unsigned int iters )
    {
      NumberOfIterations = iters;
    }

  /** After the specified number of iterations have been executed, the
   *  user can get the current best objective function value with this
   *  method call */
  double GetOptimalValue()
    {
      return OptimalValue;
    };

  /** Set the initial simplex edge length. When the initial simplex is
   *  constructed, each vertex is toggled as having + and - the length
   *  specified */
  void SetInitialSimplexEdgeLength( double length )
    {
      InitialSimplexEdgeLength = length;
    }

  /** After the specified number of iterations have been executed, the
   *  user can get the current best objective function parameters with
   *  this method call */
  void GetOptimalParameters( double* );

  /** This method allows users to get a pointer to the thin plate
   *  spline surface contained in the metric. This is useful as it can
   *  be passed to a viewing utility to visualize the surface after
   *  optimizer convergence */
  /* cipThinPlateSplineSurface* GetThinPlateSplineSurface() */
  /*   { */
  /*     return Metric->GetThinPlateSplineSurface(); */
  /*   } */

  /** Calling this method starts the optimization */
  void Update();

private:
  //
  // The following structure facilitates simplex vertex book-keeping.
  //
  struct SIMPLEXVERTEX
  {
    double                 value; // Value at current vertex location
    std::vector< double >  coordinates; // Location in parameter space
                                        // for current vertex
    unsigned int           rank;  // Value ranking wrt other vertices
  };
  
  void InitializeSimplex();
  void UpdateRankings();

  std::vector< SIMPLEXVERTEX > SimplexVertices;

  unsigned int Dimension;
  unsigned int BestIndex;
  unsigned int WorstIndex;
  unsigned int NumberOfIterations;

  double OptimalValue;
  double InitialSimplexEdgeLength;
  double BestValue;
  double WorstValue;
  double WorstRunnerUpValue;

  cipThinPlateSplineSurfaceModelToParticlesMetric* Metric;
  double* InitialParams;
  double* OptimalParams;
};

#endif
