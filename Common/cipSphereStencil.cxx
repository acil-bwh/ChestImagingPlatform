/**
 *
 *  $Date: 2012-06-04 17:08:46 -0700 (Mon, 04 Jun 2012) $
 *  $Revision: 137 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#include <math.h>
#include "cipSphereStencil.h"


cipSphereStencil::cipSphereStencil()
{
  this->Radius = 1.0;
}

bool cipSphereStencil::IsInsideBoundingBox( double x, double y, double z ) const
{
  if ( x >= this->BoundingBoxMin[0] && x <= this->BoundingBoxMax[0] &&
       y >= this->BoundingBoxMin[1] && y <= this->BoundingBoxMax[1] &&
       z >= this->BoundingBoxMin[2] && z <= this->BoundingBoxMax[2] )
    {
    return true;
    }

  return false;
}


bool cipSphereStencil::IsInsideStencilPattern( double x, double y, double z ) const
{
  if ( pow( x-this->Center[0], 2 ) + pow( y-this->Center[1], 2 ) + 
       pow( z-this->Center[2], 2 ) <= pow( this->Radius, 2 ) )
    {
    return true;
    }

  return false;
}


void cipSphereStencil::GetStencilBoundingBox( double* const bbMin, double* const bbMax ) const
{
  bbMin[0] = this->BoundingBoxMin[0];
  bbMin[1] = this->BoundingBoxMin[1];
  bbMin[2] = this->BoundingBoxMin[2];

  bbMax[0] = this->BoundingBoxMax[0];
  bbMax[1] = this->BoundingBoxMax[1];
  bbMax[2] = this->BoundingBoxMax[2];
}


void cipSphereStencil::SetCenter( double x, double y, double z )
{
  this->Center[0] = x;
  this->Center[1] = y;
  this->Center[2] = z;

  this->ComputeStencilBoundingBox();
}


//
// Assumes 'Center' has been set
//
void cipSphereStencil::ComputeStencilBoundingBox()
{
  this->BoundingBoxMin[0] = this->Center[0] - this->Radius/2.0;
  this->BoundingBoxMin[1] = this->Center[1] - this->Radius/2.0;
  this->BoundingBoxMin[2] = this->Center[2] - this->Radius/2.0;

  this->BoundingBoxMax[0] = this->Center[0] + this->Radius/2.0;
  this->BoundingBoxMax[1] = this->Center[1] + this->Radius/2.0;
  this->BoundingBoxMax[2] = this->Center[2] + this->Radius/2.0;
}
