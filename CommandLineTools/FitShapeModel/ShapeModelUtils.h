#ifndef _ShapeModelUtils_h_
#define _ShapeModelUtils_h_

#include <string>

std::string int2str( int i );

bool iequals( const std::string& a, const std::string& b );

void normalize( double* n );

void perturbNormal( double* n, unsigned int seed );

#endif //_ShapeModelUtils_h_