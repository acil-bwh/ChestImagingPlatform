/**
 *
 *  $Date: 2012-09-05 18:37:42 -0400 (Wed, 05 Sep 2012) $
 *  $Revision: 235 $
 *  $Author: jross $
 *
 */

#ifndef _cipParticleToThinPlateSplineSurfaceMetric_cxx
#define _cipParticleToThinPlateSplineSurfaceMetric_cxx

#include "cipParticleToThinPlateSplineSurfaceMetric.h"
#include "itkNumericTraits.h"

//
// Set the TPS surface that will be used for metric computation
//
void cipParticleToThinPlateSplineSurfaceMetric::SetThinPlateSplineSurface( const cipThinPlateSplineSurface& tpsSurface )
{
  this->ThinPlateSplineSurface = tpsSurface;
}


void cipParticleToThinPlateSplineSurfaceMetric::SetParticle( cip::PointType position )
{
  this->ParticlePosition[0] = position[0];
  this->ParticlePosition[1] = position[1];
  this->ParticlePosition[2] = position[2];
}


double cipParticleToThinPlateSplineSurfaceMetric::GetValue( PointType* params ) const
{
  //
  // Compute the point on the surface, 's', given the params (domain
  // location) 
  //
  double s[3];
    s[0] = (*params)[0];
    s[1] = (*params)[1];
    s[2] = this->ThinPlateSplineSurface.GetSurfaceHeight( s[0], s[1] );

  double value = std::pow(this->ParticlePosition[0]-s[0],2) + std::pow(this->ParticlePosition[1]-s[1],2) + 
    std::pow(this->ParticlePosition[2]-s[2],2);

  return value;
}


double cipParticleToThinPlateSplineSurfaceMetric::GetValueAndGradient( PointType* params, VectorType* gradient ) const
{
  //
  // Define 'p' to hold the particle's position for notational
  // readability  
  //
  cip::PointType p(3);    
    p[0] = this->ParticlePosition[0];
    p[1] = this->ParticlePosition[1];
    p[2] = this->ParticlePosition[2];

  //
  // Compute the point on the surface, 's', given the params (domain
  // location) 
  //
  cip::PointType s(3);
    s[0] = (*params)[0];
    s[1] = (*params)[1];
    s[2] = this->ThinPlateSplineSurface.GetSurfaceHeight( s[0], s[1] );

  double value = std::pow(s[0]-p[0],2) + std::pow(s[1]-p[1],2) + std::pow(s[2]-p[2],2);

  cip::VectorType n(3);
  this->ThinPlateSplineSurface.GetNonNormalizedSurfaceNormal( s[0], s[1], n );

  (*gradient)[0] = 2.0*(s[0] - p[0] - n[0]*(s[2]-p[2])); 
  (*gradient)[1] = 2.0*(s[1] - p[1] - n[1]*(s[2]-p[2]));  

  return value;
}


double cipParticleToThinPlateSplineSurfaceMetric::GetValueGradientAndHessian( PointType* params, VectorType* gradient, MatrixType* hessian ) const
{
  //
  // Define 'p' to hold the particle's position for notational
  // readability  
  //
  cip::PointType p(3);    
    p[0] = this->ParticlePosition[0];
    p[1] = this->ParticlePosition[1];
    p[2] = this->ParticlePosition[2];

  //
  // Compute the point on the surface, 's', given the params (domain
  // location) 
  //
  cip::PointType s(3);
    s[0] = (*params)[0];
    s[1] = (*params)[1];
    s[2] = this->ThinPlateSplineSurface.GetSurfaceHeight( s[0], s[1] );

  double value = std::pow(s[0]-p[0],2) + std::pow(s[1]-p[1],2) + std::pow(s[2]-p[2],2);

  //
  // Compute the gradient
  //
  cip::VectorType n(3);
  this->ThinPlateSplineSurface.GetNonNormalizedSurfaceNormal( s[0], s[1], n );

  (*gradient)[0] = 2.0*(s[0] - p[0] - n[0]*(s[2]-p[2])); 
  (*gradient)[1] = 2.0*(s[1] - p[1] - n[1]*(s[2]-p[2]));  

  //
  // Compute the Hessian. 'w' and 'surfPoints' are quantities
  // associated with the TPS surface, and they are needed for the
  // Hessian computation
  //
  const std::vector< double > w = this->ThinPlateSplineSurface.GetWVector();
  const std::vector< cip::PointType > surfPoints = this->ThinPlateSplineSurface.GetSurfacePoints();

  double r, drdx, drdy;
  double d3dx   = 0.0;
  double d14dy  = 0.0;
  double d19dy  = 0.0;
  for ( unsigned int i=0; i<w.size(); i++ )
    {
    double xDiffi = s[0] - surfPoints[i][0];
    double yDiffi = s[1] - surfPoints[i][1];

    r    = std::sqrt( std::pow(xDiffi,2) + std::pow(yDiffi,2) );
    drdx = xDiffi/r;
    drdy = yDiffi/r;

    double rln10   = r*vnl_math::ln10;
    double rGroup  = 2.0*std::log10(r) + 1.0/vnl_math::ln10;
    double rrGroup = r*rGroup;

    double d11dx  = (r-drdx*xDiffi)/std::pow(r,2);
    double d16dy  = (r-drdy*yDiffi)/std::pow(r,2);

    double d10dx  = drdx*((2.0/rln10) + rGroup);
    double d10dy  = 2.0*(yDiffi/(rln10)) + (yDiffi/r)*rGroup;

    double d11dy = -xDiffi*(yDiffi/r)/std::pow(r,2);

    d19dy -= w[i]*( rrGroup*d11dy + drdx*d10dy );
    d3dx  -= w[i]*( rrGroup*d11dx + drdx*d10dx );
    d14dy -= w[i]*( rrGroup*d16dy + (yDiffi/r)*d10dy );
    }

  double zDiff = (s[2]-p[2]);

  double d2gdx2  =  2.0*(1.0 + n[0]*n[0] - zDiff*d3dx );
  double d2gdy2  =  2.0*(1.0 + n[1]*n[1] - zDiff*d14dy );
  double d2gdydx = -2.0*(-n[0]*n[1] + zDiff*d19dy);

  (*hessian)[0][0] = d2gdx2;
  (*hessian)[1][1] = d2gdy2;
  (*hessian)[0][1] = d2gdydx;
  (*hessian)[1][0] = d2gdydx;

  return value;
}


double cipParticleToThinPlateSplineSurfaceMetric::GetAngleBetweenVectors( const double vec1[3], const double vec2[3] ) const
{
  double vec1Mag = this->GetVectorMagnitude( vec1 );
  double vec2Mag = this->GetVectorMagnitude( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( std::abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = std::acos( arg );

  return angle;   
}


double cipParticleToThinPlateSplineSurfaceMetric::GetVectorMagnitude( const double vector[3] ) const
{
  double magnitude = std::sqrt( std::pow( vector[0], 2 ) + std::pow( vector[1], 2 ) + std::pow( vector[2], 2 ) );

  return magnitude;
}

#endif
