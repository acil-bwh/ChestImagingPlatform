#include "cipChestRegionChestTypeLocationsIO.h"
#include <algorithm>

int main( int argc, char* argv[] )
{
  // Establish ground truth
  std::vector< unsigned char > gtRegions;
  std::vector< unsigned char > gtTypes;
  std::vector< double* >       gtPoints;

  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {117.071, -205.229, -261}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {117.071, -198.977, -238}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {117.071, -179.653, -207.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {117.071, -154.077, -180.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {117.071, -120.544, -147.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {107.409, -214.323, -276}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {107.409, -196.136, -238}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {107.409, -163.171, -190.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {107.409, -118.271, -144.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {107.409, -100.651, -128}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -200.114, -255.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -181.927, -232.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -177.948, -223.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -127.933, -149.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -105.767, -127.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {88.0846, -83.6006, -105}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -189.315, -247.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -176.811, -228}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -130.206, -155.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -125.659, -143}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -106.903, -127}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {78.4225, -76.2119, -95.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {67.6237, -123.954, -141}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {67.6237, -101.22, -117}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {67.6237, -73.3701, -87.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {57.3932, -126.796, -144}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {57.3932, -112.019, -127.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {57.3932, -90.9893, -103.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {57.3932, -75.0752, -81.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {45.4577, -126.228, -137.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {45.4577, -119.407, -129}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {45.4577, -97.2412, -108.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {45.4577, -75.6436, -81}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {36.9323, -123.954, -126}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {36.9323, -105.767, -116}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {36.9323, -88.1475, -95}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {36.9323, -77.917, -81}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {25.5651, -105.198, -108.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {25.5651, -95.5361, -96}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {25.5651, -85.874, -82.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {13.0612, -109.745, -104}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {13.0612, -104.63, -98}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 3 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {13.0612, -100.651, -92}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-113.683, -181.358, -188}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-113.683, -163.171, -172}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-113.683, -141.005, -155}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-113.683, -118.839, -140.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 0 );
  gtTypes.push_back( 0 );
  {double tmp[3] = {-113.683, -108.04, -136.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-104.021, -202.388, -210}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-104.021, -176.243, -184.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-104.021, -143.847, -156.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-104.021, -118.839, -138}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-104.021, -93.2627, -127}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -220.007, -244.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -200.683, -216}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -160.329, -174.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -125.659, -143}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -98.3779, -125.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-86.9699, -75.0752, -113}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-74.466, -218.302, -247.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-74.466, -166.013, -184}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-74.466, -121.681, -139.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-74.466, -83.6006, -115.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-74.466, -69.96, -107.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -204.661, -237}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -178.517, -204}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -148.962, -170}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -128.501, -145.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -106.903, -128}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -81.3272, -113.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-64.2356, -67.6866, -103}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -204.093, -235}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -175.106, -201.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -150.099, -173.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -135.321, -151}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -117.702, -132}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -90.4209, -117}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -72.8018, -108}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-55.7102, -68.2549, -101}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -201.251, -234}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -180.79, -214.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -158.624, -180}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -119.407, -131}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -100.651, -118}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -83.0322, -110.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-43.7747, -72.2334, -104.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-34.1126, -174.538, -204.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-34.1126, -163.171, -185}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-112.546, -151.235, -116}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-112.546, -137.026, -122.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-112.546, -119.976, -140}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-105.157, -169.423, -111}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-105.157, -150.099, -117}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-105.157, -139.3, -122.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-105.157, -120.544, -139}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-96.0637, -179.653, -107}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-96.0637, -150.667, -119.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-96.0637, -143.278, -122}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 8 );
  {double tmp[3] = {-96.0637, -123.954, -141}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-86.9699, -186.474, -104.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-86.9699, -176.811, -107}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-86.9699, -162.034, -119.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-86.9699, -133.616, -139}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-86.9699, -127.364, -143}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-91.5168, -182.495, -105.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-91.5168, -161.466, -117}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-91.5168, -143.847, -122.5}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-91.5168, -129.069, -138}; gtPoints.push_back( tmp );}
  gtRegions.push_back( 2 );
  gtTypes.push_back( 9 );
  {double tmp[3] = {-91.5168, -125.091, -141.5}; gtPoints.push_back( tmp );}
    
  std::cout << "Reading region and type points file..." << std::endl;
  cipChestRegionChestTypeLocationsIO regionsTypesIO;
    regionsTypesIO.SetFileName( argv[1] );
    regionsTypesIO.Read();

  for ( unsigned int i=0; i<regionsTypesIO.GetOutput()->GetNumberOfTuples(); i++ )
    {      
      unsigned char cipRegion = regionsTypesIO.GetOutput()->GetChestRegionValue( i );
      unsigned char cipType   = regionsTypesIO.GetOutput()->GetChestTypeValue( i );
      double* point = new double[3];
      regionsTypesIO.GetOutput()->GetLocation( i, point );

      if ( cipRegion != gtRegions[i] || cipType != gtTypes[i] || gtPoints[i][0] != point[0] || 
	   gtPoints[i][1] != point[1] || gtPoints[i][2] != point[2] )
	{
	  std::cout << "FAILED" << std::endl;
	  return 1;
	}
      delete[] point;
    }
  
  std::cout << "PASSED" << std::endl;
  return 0;
}
