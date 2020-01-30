/**
 *
 *  $Date: 2012-06-08 13:33:39 -0700 (Fri, 08 Jun 2012) $
 *  $Revision: 148 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#include <iostream>
#include "vnl/vnl_math.h"
#include <cfloat>
#include "cipCylinderStencil.h"


cipCylinderStencil::cipCylinderStencil()
{
  this->Radius = 1.0;
  this->Height = 1.0;
  this->Orientation = new double[3];
  this->Orientation[0] = 0;
  this->Orientation[1] = 0;
  this->Orientation[2] = 1;
}


cipCylinderStencil::~cipCylinderStencil()
{
  delete this->Orientation;
}


bool cipCylinderStencil::IsInsideBoundingBox( double x, double y, double z ) const
{
  if ( x >= this->BoundingBoxMin[0] && x <= this->BoundingBoxMax[0] &&
       y >= this->BoundingBoxMin[1] && y <= this->BoundingBoxMax[1] &&
       z >= this->BoundingBoxMin[2] && z <= this->BoundingBoxMax[2] )
    {
    return true;
    }

  return false;
}


bool cipCylinderStencil::IsInsideStencilPattern( double x, double y, double z ) const
{
  //
  // Construct the position vector
  //
  double vec[3];
    vec[0] = x - this->Center[0];
    vec[1] = y - this->Center[1];
    vec[2] = z - this->Center[2];

  double mag = this->GetVectorMagnitude3D( vec );

  //
  // Now get the angle between this vector and the orientation
  // vector 
  //
  double theta = this->GetAngleBetweenVectors( vec, this->Orientation, false );

  //
  // The point is inside the cylinder provided that the projection of
  // the position vector onto the orientation vector has magnitude
  // less than or equal to halft the height, and provided that the
  // projection of the position vector onto the radial vector has a
  // magnitude less than or equal to the radius
  //
  if ( cos( theta )*mag <= this->Height/2.0 && sin( theta )*mag <= this->Radius )
    {
    return true;
    }

  return false;
}


void cipCylinderStencil::GetStencilBoundingBox( double* const bbMin, double* const bbMax ) const
{
  bbMin[0] = this->BoundingBoxMin[0];
  bbMin[1] = this->BoundingBoxMin[1];
  bbMin[2] = this->BoundingBoxMin[2];

  bbMax[0] = this->BoundingBoxMax[0];
  bbMax[1] = this->BoundingBoxMax[1];
  bbMax[2] = this->BoundingBoxMax[2];
}


void cipCylinderStencil::SetCenter( double x, double y, double z )
{
  this->Center[0] = x;
  this->Center[1] = y;
  this->Center[2] = z;

  this->ComputeStencilBoundingBox();
}


void cipCylinderStencil::SetRadius( double r )
{
  this->Radius = r;
  this->ComputeStencilBoundingBox();
}


void cipCylinderStencil::SetHeight( double h )
{
  this->Height = h;
  this->ComputeStencilBoundingBox();
}


//
// The idea behind the bounding box calculation here is to project the
// cylinder into each of the three orthogonal planes and to perform
// geometry / bounding-box tests there.
//
void cipCylinderStencil::ComputeStencilBoundingBox()
{
  //
  // Initialize the bounding box
  //
  this->BoundingBoxMin[0] = DBL_MAX;
  this->BoundingBoxMin[1] = DBL_MAX;
  this->BoundingBoxMin[2] = DBL_MAX;

  this->BoundingBoxMax[0] = -DBL_MAX;
  this->BoundingBoxMax[1] = -DBL_MAX;
  this->BoundingBoxMax[2] = -DBL_MAX;

  //
  // First make sure that the orientation vector has magnitude equal
  // to one half the height
  //
  double mag = this->GetVectorMagnitude3D( this->Orientation );

  this->Orientation[0] = this->Height*this->Orientation[0]/(2.0*mag);
  this->Orientation[1] = this->Height*this->Orientation[1]/(2.0*mag);
  this->Orientation[2] = this->Height*this->Orientation[2]/(2.0*mag);

  //
  // Declare some temp containers that we'll need
  //
  double orientation2D[3];
  double radius2D[2]; //A vector pointing in the radial direction in
                      //the projected plane  
  double corner2D[2]; //Corners of the projected stencil. These will
                      //help us keep track of the mins and maxes in
                      //the projected planes

  //
  // Project onto x-y plane
  //
  orientation2D[0] = this->Center[0] + this->Orientation[0];
  orientation2D[1] = this->Center[1] + this->Orientation[1];

  radius2D[0] = -this->Orientation[1];
  radius2D[1] =  this->Orientation[0];
  if ( radius2D[0] == 0 && radius2D[1] == 0 )
    {
    radius2D[0] = std::sqrt(2.0)*this->Radius;
    }
  else
    {
    mag = this->GetVectorMagnitude2D( radius2D );
    radius2D[0] = std::sqrt(2.0)*this->Radius*radius2D[0]/mag;
    radius2D[1] = std::sqrt(2.0)*this->Radius*radius2D[1]/mag;
    }

  //
  // There are four corners of the stencil pattern in the projected 2D
  // plane. We can get the corners by considering the orientation2D
  // vector and the radius2D vector, considering all four summation
  // combinations of these two vectors
  //
  for ( double i=-1.0; i<=1.0; i += 2.0 )
    {
    for ( double j=-1.0; j<=1.0; j += 2.0 )
      {
      corner2D[0] = this->Center[0] + i*this->Orientation[0] + j*radius2D[0];
      corner2D[1] = this->Center[1] + i*this->Orientation[1] + j*radius2D[1];

      corner2D[0] < this->BoundingBoxMin[0] ? this->BoundingBoxMin[0] = corner2D[0] : false;
      corner2D[1] < this->BoundingBoxMin[1] ? this->BoundingBoxMin[1] = corner2D[1] : false;

      corner2D[0] > this->BoundingBoxMax[0] ? this->BoundingBoxMax[0] = corner2D[0] : false;
      corner2D[1] > this->BoundingBoxMax[1] ? this->BoundingBoxMax[1] = corner2D[1] : false;
      }
    }

  //
  // Project onto x-z plane
  //
  orientation2D[0] = this->Center[0] + this->Orientation[0];
  orientation2D[1] = this->Center[2] + this->Orientation[2];

  radius2D[0] = -this->Orientation[2];
  radius2D[1] =  this->Orientation[0];
  if ( radius2D[0] == 0 && radius2D[1] == 0 )
    {
    radius2D[0] = std::sqrt(2.0)*this->Radius;
    }
  else
    {
    mag = this->GetVectorMagnitude2D( radius2D );
    radius2D[0] = std::sqrt(2.0)*this->Radius*radius2D[0]/mag;
    radius2D[1] = std::sqrt(2.0)*this->Radius*radius2D[1]/mag;
    }

  //
  // There are four corners of the stencil pattern in the projected 2D
  // plane. We can get the corners by considering the orientation2D
  // vector and the radius2D vector, considering all four summation
  // combinations of these two vectors
  //
  for ( double i=-1.0; i<=1.0; i += 2.0 )
    {
    for ( double j=-1.0; j<=1.0; j += 2.0 )
      {
      corner2D[0] = this->Center[0] + i*this->Orientation[0] + j*radius2D[0];
      corner2D[1] = this->Center[2] + i*this->Orientation[2] + j*radius2D[1];

      corner2D[0] < this->BoundingBoxMin[0] ? this->BoundingBoxMin[0] = corner2D[0] : false;
      corner2D[1] < this->BoundingBoxMin[2] ? this->BoundingBoxMin[2] = corner2D[1] : false;

      corner2D[0] > this->BoundingBoxMax[0] ? this->BoundingBoxMax[0] = corner2D[0] : false;
      corner2D[1] > this->BoundingBoxMax[2] ? this->BoundingBoxMax[2] = corner2D[1] : false;
      }
    }

  //
  // Project onto y-z plane
  //
  orientation2D[0] = this->Center[1] + this->Orientation[1];
  orientation2D[1] = this->Center[2] + this->Orientation[2];

  radius2D[0] = -this->Orientation[2];
  radius2D[1] =  this->Orientation[1];
  if ( radius2D[0] == 0 && radius2D[1] == 0 )
    {
    radius2D[0] = std::sqrt(2.0)*this->Radius;
    }
  else
    {
    mag = this->GetVectorMagnitude2D( radius2D );
    radius2D[0] = std::sqrt(2.0)*this->Radius*radius2D[0]/mag;
    radius2D[1] = std::sqrt(2.0)*this->Radius*radius2D[1]/mag;
    }

  //
  // There are four corners of the stencil pattern in the projected 2D
  // plane. We can get the corners by considering the orientation2D
  // vector and the radius2D vector, considering all four summation
  // combinations of these two vectors
  //
  for ( double i=-1.0; i<=1.0; i += 2.0 )
    {
    for ( double j=-1.0; j<=1.0; j += 2.0 )
      {
      corner2D[0] = this->Center[1] + i*this->Orientation[1] + j*radius2D[0];
      corner2D[1] = this->Center[2] + i*this->Orientation[2] + j*radius2D[1];

      corner2D[0] < this->BoundingBoxMin[1] ? this->BoundingBoxMin[1] = corner2D[0] : false;
      corner2D[1] < this->BoundingBoxMin[2] ? this->BoundingBoxMin[2] = corner2D[1] : false;

      corner2D[0] > this->BoundingBoxMax[1] ? this->BoundingBoxMax[1] = corner2D[0] : false;
      corner2D[1] > this->BoundingBoxMax[2] ? this->BoundingBoxMax[2] = corner2D[1] : false;
      }
    }
}


void cipCylinderStencil::SetOrientation( double x, double y, double z )
{
  this->Orientation[0] = x;
  this->Orientation[1] = y;
  this->Orientation[2] = z;

  this->ComputeStencilBoundingBox();
}


double cipCylinderStencil::GetVectorMagnitude3D( double vec[3] ) const
{
  return sqrt( pow( vec[0], 2 ) + pow( vec[1], 2 ) + pow( vec[2], 2 ) );
}


double cipCylinderStencil::GetVectorMagnitude2D( double vec[2] ) const
{
  return sqrt( pow( vec[0], 2 ) + pow( vec[1], 2 ) );  
}


double cipCylinderStencil::GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees ) const
{
  double vec1Mag = this->GetVectorMagnitude3D( vec1 );
  double vec2Mag = this->GetVectorMagnitude3D( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( std::abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = acos( arg );

  if ( !returnDegrees )
    {
    if ( angle > vnl_math::pi/2.0 )
      {
      return vnl_math::pi - angle;
      }

    return angle;
    }

  double angleInDegrees = (180.0/vnl_math::pi)*angle;

  if ( angleInDegrees > 90.0 )
    {
    angleInDegrees = 180.0 - angleInDegrees;
    }

  return angleInDegrees;
}
