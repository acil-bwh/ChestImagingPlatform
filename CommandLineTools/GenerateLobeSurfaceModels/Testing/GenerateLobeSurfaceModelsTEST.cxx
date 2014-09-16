#include "GenerateLobeSurfaceModelsHelper.h"
#include <cmath>

int main( int argc, char* argv[] )
{
  double eps = 1e-5;

  std::vector< double > sample1;  sample1.push_back( 2.94172302 );  sample1.push_back( 3.75132945 );
  std::vector< double > sample2;  sample2.push_back( -1.36141897 ); sample2.push_back( -2.61915722 );
  std::vector< double > sample3;  sample3.push_back( -0.53473764 ); sample3.push_back( -1.66211294 );
  std::vector< double > sample4;  sample4.push_back( 1.66883093 );  sample4.push_back(  2.02706592 );
  std::vector< double > sample5;  sample5.push_back( -3.15739708 ); sample5.push_back( -3.54165843 );
  std::vector< double > sample6;  sample6.push_back( -0.89478389 ); sample6.push_back(  0.03591439 );
  std::vector< double > sample7;  sample7.push_back( -0.94983085 ); sample7.push_back( -0.56290266 );
  std::vector< double > sample8;  sample8.push_back( -1.86802405 ); sample8.push_back( -1.971469   );
  std::vector< double > sample9;  sample9.push_back( 2.12955876 );  sample9.push_back(  2.57488365 );
  std::vector< double > sample10; sample10.push_back( 2.02607978 ); sample10.push_back(  1.96810682 );

  std::vector< std::vector< double > > data;
    data.push_back( sample1 );
    data.push_back( sample2 );
    data.push_back( sample3 );
    data.push_back( sample4 );
    data.push_back( sample5 );
    data.push_back( sample6 );
    data.push_back( sample7 );
    data.push_back( sample8 );
    data.push_back( sample9 );
    data.push_back( sample10 );

  PCA dataPCA = GetLowDimensionPCA( data );

  std::vector< double > gtMean; gtMean.push_back(9.99999994e-10); gtMean.push_back(-2.00000003e-09);

  for ( unsigned int i=0; i<dataPCA.meanVec.size(); i++ )
    {
      if ( std::abs( gtMean[i] - dataPCA.meanVec[i] ) > eps )
	{
	  std::cout << "FAILED" << std::endl;
	  return 1;
	}
    }

  if ( dataPCA.numModes != 2 )
    {
      std::cout << "FAILED" << std::endl;
      return 1;
    }

  std::vector< double > gtModes; gtModes.push_back( 9.08135962 ); gtModes.push_back( 0.16217744 );

  for ( unsigned int i=0; i<dataPCA.numModes; i++ )
    {
      if ( std::abs( gtModes[i] - dataPCA.modeVec[i] ) > eps )
	{
	  std::cout << "FAILED" << std::endl;
	  return 1;
	} 
    }

  std::vector< std::vector< double > > gtEigVecs;
  std::vector< double > gtEigVec1; gtEigVec1.push_back( 0.63317433 ); gtEigVec1.push_back( 0.77400922 );
  std::vector< double > gtEigVec2; gtEigVec2.push_back( 0.77400922 ); gtEigVec2.push_back( -0.63317433 );
  gtEigVecs.push_back( gtEigVec1 ); 
  gtEigVecs.push_back( gtEigVec2 );

  for ( unsigned int i=0; i<dataPCA.numModes; i++ )
    {
      for ( unsigned int j=0; j<dataPCA.numModes; j++ )
	{
	  if ( std::abs( gtEigVecs[i][j] - dataPCA.modeVecVec[i][j] ) > eps )
	    {
	      std::cout << "FAILED" << std::endl;
	      return 1;
	    }
	}
    }

  std::cout << "PASSED" << std::endl;
  return 0;
}
