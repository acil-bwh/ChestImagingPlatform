/**
 *  \file cipSphereStencil
 *  \ingroup common
 *  \brief This class implements the sphere stencil. Users have
 *  control over the radius.
 *
 *  $Date: 2012-06-06 15:25:53 -0700 (Wed, 06 Jun 2012) $
 *  $Revision: 146 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#ifndef __cipSphereStencil_h
#define __cipSphereStencil_h

#include "cipStencil.h"

class cipSphereStencil : public cipStencil
{
public:
  ~cipSphereStencil(){};
  cipSphereStencil();

  /** Given physical coordinates, x, y, and z, this method will
   *  indicate whether the point is inside the stencil's bounding box
   *  or not. Note that 'SetCenter' must be called before calling this
   *  method. */
  bool IsInsideBoundingBox( double, double, double ) const;

  /** Given physical coordinates, x, y, and z, this method will
   *  indicate whether the point is inside the stencil pattern or
   *  not. Note that 'SetCenter' must be called before calling this
   *  method. Note that 'SetCenter' must be called before calling this
   *  method. */
  bool IsInsideStencilPattern( double, double, double ) const; 

  /** Get the bounding box of the stencil. The first argument should
   *  be a 3 element vector to hold the min x, y, and z physical
   *  coordinates of the bounding box. The second argument should be
   *  a 3 element vector to hold the max x, y, and z physical
   *  coordinates of the bounding box. Note that 'SetCenter' must be
   *  called before calling this method. */
  void GetStencilBoundingBox( double* const, double* const ) const;

  /** Set the physical point (x, y, and z coordiantes) at which the
   *  stencil will be created. */
  void SetCenter( double, double, double );

  /** Set the x, y, and z values of the orientation vector. Calling
   *  this method has no effect for this stencil, but must be defined
   *  given inheritance from cipStencil */
  void SetOrientation( double, double, double ){};

  /** Set the radius of the sphere stencil */
  void SetRadius( double r )
    {
      Radius = r;
    };

private:
  void ComputeStencilBoundingBox();

  double Radius;
};

#endif
