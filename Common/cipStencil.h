/**
 *  \file cipSphereStencil
 *  \ingroup common
 *  \brief This class is an abstract base class for stencil
 *  patterns. It declares core methods that should exist across all
 *  stencil types. The basic idea behind a stencil is that you supply
 *  a single physical point, and the stencil class centers a pattern
 *  at that point (a sphere, an oriented cylinder, etc). Once a point
 *  is supplied, you can ask whether or not any other point is within
 *  the pattern or within the axis-aligned bounding box of the
 *  pattern. Everything is done in physical space.
 *
 *  $Date: 2012-06-11 17:59:03 -0700 (Mon, 11 Jun 2012) $
 *  $Revision: 157 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#ifndef __cipStencil_h
#define __cipStencil_h

#include "cipStencil.h"

class cipStencil
{
public:
  ~cipStencil(){};
  cipStencil(){};

  /** Given physical coordinates, x, y, and z, this method will
   *  indicate whether the point is inside the stencil's bounding box
   *  or not. Note that 'SetCenter' must be called before calling this
   *  method. */
  virtual bool IsInsideBoundingBox( double, double, double ) const = 0;

  /** Given physical coordinates, x, y, and z, this method will
   *  indicate whether the point is inside the stencil pattern or
   *  not. Note that 'SetCenter' must be called before calling this
   *  method. */
  virtual bool IsInsideStencilPattern( double, double, double ) const = 0;

  /** Get the bounding box of the stencil. The first argument should
   *  be a 3 element vector to hold the min x, y, and z physical
   *  coordinates of the bounding box. The second argument should be
   *  a 3 element vector to hold the max x, y, and z physical
   *  coordinates of the bounding box. Note that 'SetCenter' must be
   *  called before calling this method. */
  virtual void GetStencilBoundingBox( double* const, double* const ) const = 0;

  /** Set the physical point (x, y, and z coordiantes) at which the
   *  stencil will created. Subclasses should set the center and then
   *  immediately call 'ComputeStencilBoundingBox' */
  virtual void SetCenter( double, double, double ) = 0;

  /** xSet the orientation of the stencil. Parameters are the x,
   *  y, and z values of the orientation vector */
  virtual void SetOrientation( double, double, double ) = 0;

  /** Set the radius of the stencil pattern */
  virtual void SetRadius( double ) = 0;

protected:
  virtual void ComputeStencilBoundingBox() = 0;

  double Center[3];
  double BoundingBoxMin[3];
  double BoundingBoxMax[3];
};

#endif
