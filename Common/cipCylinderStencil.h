/**
 *  \file cipCylinderStencil
 *  \ingroup common
 *  \brief This class implements the cylinder stencil. Users have
 *  control over the radius and height of the cylinder. A 3D vector
 *  indicating the cylinder's alignment must also be supplied.
 *
 *  $Date: 2012-06-08 13:33:39 -0700 (Fri, 08 Jun 2012) $
 *  $Revision: 148 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#ifndef __cipCylinderStencil_h
#define __cipCylinderStencil_h

#include "cipStencil.h"

class cipCylinderStencil : public cipStencil
{
public:
  ~cipCylinderStencil();
  cipCylinderStencil();

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

  /** Set the radius of the sphere stencil */
  void SetRadius( double );

  /** Set the height of the sphere stencil */
  void SetHeight( double );

  /** Set orientation vector */
  void SetOrientation( double, double, double );  

private:
  void   ComputeStencilBoundingBox();
  double GetVectorMagnitude2D( double[2] ) const;
  double GetVectorMagnitude3D( double[3] ) const;
  double GetAngleBetweenVectors( double[3], double[3], bool ) const;

  double  Radius;
  double  Height;
  double* Orientation;
};

#endif
