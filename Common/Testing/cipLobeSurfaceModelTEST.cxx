#include "cipLobeSurfaceModel.h"
#include "cipLobeSurfaceModelIO.h"
#include "cipExceptionObject.h"
#include <iostream>
#include <cmath>

int main( int argc, char* argv[] )
{
  std::cout << "Reading lobe surface model..." << std::endl;
  cip::LobeSurfaceModelIO reader;
    reader.SetFileName( argv[1] );
  try
    {
    reader.Read();
    }
  catch ( cip::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading lobe surface model:";
    std::cerr << excp << std::endl;
    }

  reader.GetOutput()->SetRightLungSurfaceModel( true );
  
  double eps = 1e-5;
  unsigned int numROPoints = reader.GetOutput()->GetRightObliqueWeightedSurfacePoints().size();
  unsigned int numRHPoints = reader.GetOutput()->GetRightHorizontalWeightedSurfacePoints().size();

  // First test: see if we can correctly get the weighted right oblique surface
  // points
  {    
    double roZVal1 = reader.GetOutput()->GetRightObliqueWeightedSurfacePoints()[0][2];
    double roZVal1_gt = -229.95059550345539;
    
    double roZVal2 = reader.GetOutput()->GetRightObliqueWeightedSurfacePoints()[numROPoints-1][2];
    double roZVal2_gt = -261.34376358321458;
    
    if ( std::abs(roZVal1 - roZVal1_gt) > eps || std::abs(roZVal2 - roZVal2_gt) > eps )
      {
	std::cout << "FAILED" << std::endl;
	return 1;
      }    
  }

  // Second test: see if we can correctly get the weighted right horizontal surface
  // points
  {
    double rhZVal1 = reader.GetOutput()->GetRightHorizontalWeightedSurfacePoints()[0][2];
    double rhZVal1_gt = -178.68186454367645;
    
    double rhZVal2 = reader.GetOutput()->GetRightHorizontalWeightedSurfacePoints()[numRHPoints-1][2];
    double rhZVal2_gt = -183.75202230625916;
    
    if ( std::abs(rhZVal1 - rhZVal1_gt) > eps || std::abs(rhZVal2 - rhZVal2_gt) > eps )
      {
	std::cout << "FAILED" << std::endl;
	return 1;
      }
  }

  // Third test: see if we can correctly get the mean right oblique points
  {
    double roZVal1 = reader.GetOutput()->GetMeanRightObliqueSurfacePoints()[0][2];
    double roZVal1_gt = -194.824;

    double roZVal2 = reader.GetOutput()->GetMeanRightObliqueSurfacePoints()[numROPoints-1][2];
    double roZVal2_gt = -210.685;

    if ( std::abs(roZVal1 - roZVal1_gt) > eps || std::abs(roZVal2 - roZVal2_gt) > eps )
      {
	std::cout << "FAILED" << std::endl;
	return 1;
      }
  }

  // Fourth test: see if we can correctly get the mean right horizontal points
  {
    double rhZVal1 = reader.GetOutput()->GetMeanRightHorizontalSurfacePoints()[0][2];
    double rhZVal1_gt = -163.937;

    double rhZVal2 = reader.GetOutput()->GetMeanRightHorizontalSurfacePoints()[numRHPoints-1][2];
    double rhZVal2_gt = -164.41;

    if ( std::abs(rhZVal1 - rhZVal1_gt) > eps || std::abs(rhZVal2 - rhZVal2_gt) > eps )
      {
	std::cout << "FAILED" << std::endl;
	return 1;
      }
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
