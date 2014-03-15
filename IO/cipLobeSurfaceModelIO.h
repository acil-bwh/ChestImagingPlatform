/**
 *  \class cipLobeSurfaceModelIO
 *  \ingroup io
 *  \brief This class handles the input and output of lobe boundary
 *  shape models.
 *
 *  This class handles the input and output of lobe boundary
 *  shape models. It currently assumes a comma separated value file
 *  structure.
 *
 *  Date:     $Date: 2012-08-31 20:13:29 -0400 (Fri, 31 Aug 2012) $
 *  Version:  $Revision: 223 $
 *
 *  TODO:
 *  Eventually it might be nice to have an xml-based file structure
 *  for these files. In that case we would need to update this IO.
 *  
 */

#ifndef __cipLobeSurfaceModelIO_h
#define __cipLobeSurfaceModelIO_h

#include "cipLobeSurfaceModel.h"
#include <string>

namespace cip
{
class LobeSurfaceModelIO
{
public:
  ~LobeSurfaceModelIO();
  LobeSurfaceModelIO();

  void SetFileName( std::string );

  void Read();
  void Write() const;

  /** The returned shape model intentionally left non-const because we
   *  may with to modify the mode weights of the shape model. Doing
   *  this will allow an easy read-modify-write flow */
  cipLobeSurfaceModel* GetOutput();

  /** Set the shape model to write to file */
  void SetInput( cipLobeSurfaceModel* );

private:
  cipLobeSurfaceModel* ShapeModel;

  std::string FileName;
};

} // end namespace cip

#endif // __cipLobeSurfaceModelIO_h
