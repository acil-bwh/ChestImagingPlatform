#include "cipNelderMeadSimplexOptimizer.h"
#include "cipThinPlateSplineSurfaceModelToParticlesMetric.h"
#include "cipExceptionObject.h"
#include <cmath>

class cipTestMetric: public cipThinPlateSplineSurfaceModelToParticlesMetric
{
public:
  cipTestMetric() {};
  ~cipTestMetric() {};
  
  /** A simple quadratic cost function with a global min at the zero vector */
  double GetValue( const std::vector< double >* const params )
  {
    double value = 0;
    for ( unsigned int i=0; i<(*params).size(); i++ )
      {
	value += (*params)[i]*(*params)[i];
      }
    return value;
  }

protected:
  double GetFissureTermValue() { return 0; };
  double GetAirwayTermValue() { return 0; };
  double GetVesselTermValue() { return 0; }; 
};

int main( int argc, char* argv[] )
{
  cipTestMetric* metric = new cipTestMetric();

  // Some random location 
  double* initialParams = new double[2];
    initialParams[0] = 8.2;
    initialParams[1] = -5.1;

  cipNelderMeadSimplexOptimizer* optimizer = new cipNelderMeadSimplexOptimizer( 2 );
    optimizer->SetInitialParameters( initialParams );
    optimizer->SetMetric( metric );
    optimizer->SetNumberOfIterations( 50 );
    optimizer->SetInitialSimplexEdgeLength( 5.0 );
    optimizer->Update();

  double* optimalParams = new double[2];
  optimizer->GetOptimalParameters( optimalParams );

  double eps = 1e-5;
  if ( std::abs(optimalParams[0]) > eps || std::abs(optimalParams[1]) > eps )
    {
      std::cout << "FAILED" << std::endl;
      return 1;
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}
