#include "ShapeModelUtils.h"
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <iostream>

std::string int2str( int i )
{
  std::stringstream sstr;
  sstr << i;
  return sstr.str();
}

bool iequals( const std::string& a, const std::string& b )
{
  unsigned int sz = a.size();
  if (b.size() != sz)
  {
    return false;
  }
  for (unsigned int i = 0; i < sz; ++i)
  {
    if (tolower(a[i]) != tolower(b[i]))
    {
      return false;
    }
  }
  return true;
}

void normalize( double* n )
{
  double d = sqrt( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );
  n[0] /= d;
  n[1] /= d;
  n[2] /= d;
}

void perturbNormal( double* n, unsigned int seed )
{
  if (seed == 0)
  {
    return;
  }
  
  //srand( seed );
  double f = 0.2; // perturb maximum 100*f % of the length on each axis
  
  for (int i = 0; i < 3; i++)
  {
    double r = rand() / (double)RAND_MAX - 0.5; // [-0.5..0.5]
    //std::cout << "          " << r;
    n[i] += f * r;
  }
  //std::cout << std::endl;

  normalize(n);
}