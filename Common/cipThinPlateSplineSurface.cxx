#include "cipThinPlateSplineSurface.h"
#include "itkNumericTraits.h"


cipThinPlateSplineSurface::cipThinPlateSplineSurface()
{
  this->m_Lambda = 0.0;
  this->m_NumberSurfacePoints = 0;
}


//
// This method makes a copy of the incoming points, so the pointers of
// the incoming vector can go out of scope, and we will still be safe
//
cipThinPlateSplineSurface::cipThinPlateSplineSurface( const std::vector< cip::PointType >& surfacePointsVec )
{
  this->m_NumberSurfacePoints = surfacePointsVec.size();
  this->SetSurfacePoints( surfacePointsVec );
  this->ComputeThinPlateSplineVectors();
}

void cipThinPlateSplineSurface::SetLambda( double lambda )
{
  this->m_Lambda = lambda;

  if ( this->m_NumberSurfacePoints > 0 )
    {
    this->ComputeThinPlateSplineVectors();
    }
}


void cipThinPlateSplineSurface::SetSurfacePointWeights( const std::vector< double >*  const surfacePointWeights )
{
  // Clear any existing surface point weights first
  this->m_SurfacePointWeights.clear();

  for ( unsigned int i=0; i<surfacePointWeights->size(); i++ )
    {
    this->m_SurfacePointWeights.push_back( (*surfacePointWeights)[i] );
    }
    
  // Compute the TPS vectors given these new point weights
  this->ComputeThinPlateSplineVectors();
}


void cipThinPlateSplineSurface::SetSurfacePoints( const std::vector< cip::PointType >& surfacePointsVec )
{
  // Make sure any old memory is freed up and the vector of surface
  // points is cleared before we add new points.
  this->m_SurfacePoints.clear();

  // We also assume that if new points are being added, any weights
  // previously set are now irrelevant, so we clear this container to
  // make sure they don't have an effect on the new TPS computation
  this->m_SurfacePointWeights.clear();

  // Now we can add the new points
  for ( unsigned int i=0; i<surfacePointsVec.size(); i++ )
    {
      cip::PointType point(3);
        point[0] = surfacePointsVec[i][0];
	point[1] = surfacePointsVec[i][1];
	point[2] = surfacePointsVec[i][2];

    this->m_SurfacePoints.push_back( point );
    }  

  this->m_NumberSurfacePoints = this->m_SurfacePoints.size();

  // Finally, compute the TPS vectors given these new points 
  this->ComputeThinPlateSplineVectors();
}


void cipThinPlateSplineSurface::ComputeThinPlateSplineVectors()
{
  // First make sure the TPS vectors are clear
  this->m_a.clear();
  this->m_w.clear();    

  // Create the K matrix
  double rTotal = 0.0; // Will be used to compute alpha for smoothing 

  unsigned int numPoints = this->m_SurfacePoints.size();
  
  vnl_matrix< double > K( numPoints, numPoints );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    for ( unsigned int j=0; j<numPoints; j++ )
      {
      double x1 = this->m_SurfacePoints[i][0];
      double y1 = this->m_SurfacePoints[i][1];
      double x2 = this->m_SurfacePoints[j][0];
      double y2 = this->m_SurfacePoints[j][1];

      double r = std::sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
      rTotal += r;

      if ( i == j )
        {
        K[i][j] = 0;
        }
      else
        {
	if ( r == 0.0 )
	  {
	  K[i][j] = 0.0;
	  }
	else
	  {
	  K[i][j] = r*r*std::log10( r );
	  }
        }
      }
    }
  double alpha = rTotal/static_cast< double >( numPoints*numPoints );

  // Now include the smoothing. Refer to http://elonen.iki.fi/code/tpsdemo/
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    if ( this->m_SurfacePointWeights.size() != numPoints )
      {
      K[i][i] = this->m_Lambda*alpha*alpha;
      }
    else
      {
      K[i][i] = this->m_Lambda*this->m_SurfacePointWeights[i];
      }
    }

  // Create the O matrix
  vnl_matrix< double > oMatrix( 3, 3 );
  for ( int i=0; i<3; i++ )
    {
    for ( int j=0; j<3; j++ )
      {
      oMatrix[i][j] = 0;
      }
    }

  // Create the P matrix
  vnl_matrix< double > P( numPoints, 3 );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    P[i][0] = 1;
    P[i][1] = this->m_SurfacePoints[i][0];
    P[i][2] = this->m_SurfacePoints[i][1];
    }

  // Create the L matrix
  vnl_matrix< double > L( numPoints+3, numPoints+3 );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    for ( unsigned int j=0; j<numPoints; j++ )
      {
      L[i][j] = K[i][j];
      }
    }

  for ( unsigned int i=0; i<numPoints; i++ )
    {
    for ( unsigned int j=numPoints; j<3+numPoints; j++ )
      {
      L[i][j] = P[i][j-numPoints];
      }
    }

  for ( unsigned int i=numPoints; i<(3+numPoints); i++ )
    {
    for ( unsigned int j=0; j<numPoints; j++ )
      {
      L[i][j] = P[j][i-numPoints];
      }
    }

  for ( unsigned int i=numPoints; i<(3+numPoints); i++ )
    {
    for ( unsigned int j=numPoints; j<(3+numPoints); j++ )
      {
      L[i][j] = oMatrix[i-numPoints][j-numPoints];
      }
    }

  // Create the O vector
  vnl_vector< double > oVector( 3 ); 
    oVector[0] = 0;
    oVector[1] = 0;
    oVector[2] = 0;

  // Create the v vector
  vnl_vector< double > v( numPoints );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    v[i] = this->m_SurfacePoints[i][2];
    }

  // Create the b vector, which is just the combination of v and
  // oVector
  vnl_vector< double > b( numPoints + 3 );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    b[i] = v[i];
    }
  b[numPoints]   = oVector[0];
  b[numPoints+1] = oVector[1];
  b[numPoints+2] = oVector[2];

  // We now have everything we need to solve the equation: Lx = b. b
  // is just the combination of w and a, and we'll set them explicity
  // below after we get b.  First invert L.
  vnl_matrix< double > invL = vnl_matrix_inverse< double >(L).inverse();
  vnl_vector< double > x = invL*b;

  // Now that we have x, set the w and a vectors
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    this->m_w.push_back( x[i] );
    }

  this->m_a.push_back( x[numPoints] );
  this->m_a.push_back( x[numPoints+1] );
  this->m_a.push_back( x[numPoints+2] );
}


double cipThinPlateSplineSurface::GetSurfaceHeight( double x, double y ) const
{
  unsigned int numPoints = this->m_SurfacePoints.size();

  double total = 0.0;
  for ( unsigned int n=0; n<numPoints; n++ )
    {
    double x2 = this->m_SurfacePoints[n][0];
    double y2 = this->m_SurfacePoints[n][1];
    
    double r = std::sqrt( (x-x2)*(x-x2)+(y-y2)*(y-y2) );

    if ( r!=0 )
      {
      total += this->m_w[n]*r*r*std::log10( r );
      }
    }
  double z = this->m_a[0] + x*this->m_a[1] + y*this->m_a[2] + total;

  return z;
}


void cipThinPlateSplineSurface::GetSurfaceNormal( double x, double y, cip::VectorType& normal ) const
{
  this->GetNonNormalizedSurfaceNormal( x, y, normal );

  double mag = std::sqrt( std::pow( normal[0], 2 ) + std::pow( normal[1], 2 ) + std::pow( normal[2], 2 ) );

  normal[0] = normal[0]/mag;
  normal[1] = normal[1]/mag;
  normal[2] = normal[2]/mag;
}


void cipThinPlateSplineSurface::GetNonNormalizedSurfaceNormal( double x, double y, cip::VectorType& normal ) const
{
  //
  // The normal will be computed using:
  // grad F(x,y,z) = khat - grad f(x,y)
  //
  normal[2] = 1;

  double xAccumulator = 0.0;
  double yAccumulator = 0.0;

  for ( unsigned int i=0; i<this->m_w.size(); i++ )
    {
    double xDiff = x - this->m_SurfacePoints[i][0];
    double yDiff = y - this->m_SurfacePoints[i][1];

    double r = std::sqrt( std::pow( xDiff, 2 ) + std::pow( yDiff, 2 ) );

    double dUdr = r*(2.0*std::log10( r ) + 1.0/vnl_math::ln10);
    double drdx = xDiff/r;
    double drdy = yDiff/r;

    double common = this->m_w[i]*dUdr;  // The factor common to both x
                                        // and y derivs

    xAccumulator += common*drdx;
    yAccumulator += common*drdy;
    }

  normal[0] = -( this->m_a[1] + xAccumulator );
  normal[1] = -( this->m_a[2] + yAccumulator );
}


double cipThinPlateSplineSurface::GetBendingEnergy() const
{
  // Create the K matrix
  unsigned int numPoints = this->m_SurfacePoints.size();

  vnl_matrix< double > K( numPoints, numPoints );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    for ( unsigned int j=0; j<numPoints; j++ )
      {
      double x1 = this->m_SurfacePoints[i][0];
      double y1 = this->m_SurfacePoints[i][1];
      double x2 = this->m_SurfacePoints[j][0];
      double y2 = this->m_SurfacePoints[j][1];

      double r = std::sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );

      if ( r==0 )
        {
        if ( this->m_SurfacePointWeights.size() > 0 )
          {
          if ( this->m_SurfacePointWeights[i] != 0 )
            {
            K[i][j] = 1.0/this->m_SurfacePointWeights[i];
            }
          else
            {
            K[i][j] = 1.0/itk::NumericTraits< double >::min();
            }
          }
        else
          {
          K[i][j] = 0;
          }
        }
      else
        {
        K[i][j] = r*r*std::log10( r );
        }
      }
    }

  //
  // Create a VNL vector for the weights for easy matrix
  // multiplication 
  //
  vnl_vector< double > w( numPoints );
  for ( unsigned int i=0; i<numPoints; i++ )
    {
    w[i] = this->m_w[i];
    }

  //
  // The following temp vector is created because in my 5 minutes of
  // searching, I can't figure out how to take the transpose of a vnl
  // vector!
  //
  vnl_vector< double > wK = K*w;

  double bendingEnergy = 0.0;
  for ( unsigned int i=0; i<numPoints; i++ )
    {
      bendingEnergy += w[i]*wK[i];
    }

  return bendingEnergy;
}
