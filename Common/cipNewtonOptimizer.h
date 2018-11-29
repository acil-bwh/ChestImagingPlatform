/**
 *  \file cipNewtonOptimizer
 *  \ingroup common
 *  \brief  This class is an implementation of Newton's line search 
 *  method. It is templated over the dimension of the objective 
 *  function.
 *
 *  $Date: 2012-09-05 16:59:15 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 231 $
 *  $Author: jross $
 *
 */


#ifndef __cipNewtonOptimizer_h
#define __cipNewtonOptimizer_h

#include "cipParticleToThinPlateSplineSurfaceMetric.h"
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>


template < unsigned int Dimension=2 >
class cipNewtonOptimizer
{
public:
  cipNewtonOptimizer();
  ~cipNewtonOptimizer();

  typedef vnl_vector< double >   VectorType;
  typedef vnl_vector< double >   PointType;
  typedef vnl_matrix< double >   MatrixType;

  /** This method is used to set the sufficient decrease factor that
   *  is used during the backtracking routine */
  void SetSufficientDecreaseFactor( double factor )
    {
      SufficientDecreaseFactor = factor;
    };

  /** This method is used to set the contraction factor that
   *  is used during the backtracking routine */
  void SetContractionFactor( double contractionFactor )
    {
      Rho = contractionFactor;
    }

  /** The optimizer will continue to iterate until the difference in
   *  gradient magnitude between consecutive iterations falls below 
   *  the tolerance specified with this method */ 
  void SetGradientDifferenceTolerance( double tolerance )
    {
      GradientDifferenceTolerance = tolerance;
    }

  /** Set the objective function that is to be optimized. The metric
   *  must support gradient and Hessian computations. */
  void SetMetric( const cipParticleToThinPlateSplineSurfaceMetric& m )
    {
      Metric = m;
    };

  /** Expose the metric so that it can be modified */
  cipParticleToThinPlateSplineSurfaceMetric& GetMetric()
    {
      return Metric;
    };

  /** Set the initial parameters of the objective function. */
  void SetInitialParameters( PointType* );

  double GetOptimalValue()
    {
      return OptimalValue;
    };

  void GetOptimalParameters( PointType* );

  /** Calling this method will start the optimizer */
  void Update( bool ); //DEB
  void Update();

private:
  double SufficientDecreaseFactor;  // For evaluation of sufficient decrease condition 
  double Rho; // Contraction factor
  double GradientDifferenceTolerance;  // Optimization stopping criterion
  double OptimalValue;
  
  cipParticleToThinPlateSplineSurfaceMetric Metric;
  PointType* InitialParams;
  PointType* OptimalParams;

  double LineSearch( PointType*, VectorType* );
};

#include "cipNewtonOptimizer.txx"

#endif
