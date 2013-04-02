/**
 *  \class cipThinPlateSplineSurfaceIO
 *  \ingroup io
 *  \brief This class handles the input and output of thin plate spline 
 *  surfaces.
 *
 *  Currently assumes a comma separated value file structure.
 *
 *  $Date$
 *  $Author$
 *  $Revision$
 *
 *  TODO:
 *  1) Had to change CMakeLists.txt file to link against ITK. This is due to 
 *  the as-yet-to-be-figured-out dependence that cipThinPlateSpline has on
 *  itkImage.
 */

#ifndef __cipThinPlateSplineSurfaceIO_h
#define __cipThinPlateSplineSurfaceIO_h

#include "cipThinPlateSplineSurface.h"
#include <string>

class cipThinPlateSplineSurfaceIO
{
public:
  ~cipThinPlateSplineSurfaceIO();
  cipThinPlateSplineSurfaceIO();

  void SetFileName( std::string );

  void Read();
  void Write() const;

  /**  */
  cipThinPlateSplineSurface* GetOutput();

  /** Set the shape model to write to file */
  void SetInput( cipThinPlateSplineSurface* );

private:
  cipThinPlateSplineSurface* TPS;

  std::string FileName;
};

#endif
