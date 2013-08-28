/**
 *  \class cipChestRegionChestTypeLocationsIO
 *  \ingroup io
 *  \brief This class...
 *
 *  This class...
 *
 *  $Date: 2013-04-02 12:04:01 -0400 (Tue, 02 Apr 2013) $
 *  $Revision: 399 $
 *  $Author: jross $
 *
 *  TODO:
 *  
 */

#ifndef __cipChestRegionChestTypeLocationsIO_h
#define __cipChestRegionChestTypeLocationsIO_h

#include "cipChestRegionChestTypeLocations.h"
#include <string>

class cipChestRegionChestTypeLocationsIO
{
public:
  ~cipChestRegionChestTypeLocationsIO();
  cipChestRegionChestTypeLocationsIO();
  
  /** This IO only handles one file name. However, the same IO
   *  instance can be used for both reading and writing. If separate
   *  file names are needed for reading and then writing, just use:
   *
   *  instance->SetFileName( inputFileName );
   *  instance->Read();
   *  ...
   *  instance->SetFileName( outputFileName );
   *  instance->Write(); */
  void SetFileName( std::string );

  /** Returns true if a file was found and read successfully. Note
   *  that it's generally advisable to 'Read()' before writing to
   *  attempt to load any existing data, then append new data, then
   *  write. Writing without reading first can eliminate expensive,
   *  user-defined data. */
  bool Read();

  /** It's generally advisable to attempt reading any existing
   *  indices - points file, append data as necessary, and then
   *  write. This enables a single file to contain all the region-type
   *  pairs and locations. For this reason you should almost alwasy
   *  'Read()' first before writing. Writing with an existing file
   *  name will overwrite stored data, which could cause expensive
   *  user-defined points to be lost. 'Write()' with caution. */
  void Write() const;

  /** The return type here is intentionally non-const. This is meant
   *  to make it more seemless to read, modify, then write */
  cipChestRegionChestTypeLocations* GetOutput();

  void SetInput( cipChestRegionChestTypeLocations* const );

private:
  cip::ChestConventions  Conventions;
  std::string            FileName;

  cipChestRegionChestTypeLocations* RegionTypeLocations;
};

#endif
