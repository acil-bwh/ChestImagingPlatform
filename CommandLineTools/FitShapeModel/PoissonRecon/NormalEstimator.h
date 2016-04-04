// JL: used in PoissonRecon module
#ifndef _NORMAL_ESTIMATOR_H_
#define _NORMAL_ESTIMATOR_H_

#include <vector>

#define RealType float

class NormalEstimator
{
public:
  NormalEstimator( const std::vector< RealType >& points );
  void run( std::vector< RealType >& normals );
private:
  const std::vector< RealType >& _points;
};

#endif //_NORMAL_ESTIMATOR_H_
