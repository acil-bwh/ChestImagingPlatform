/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program takes an unsigned short label map value that 
 *  conforms to the labeling conventions laid out in cipConventhions.h and 
 *  writes to the command line the corresponding chest region and chest type.
 *
 *  $Date: 2013-01-02 14:22:20 -0500 (Wed, 02 Jan 2013) $
 *  $Revision: 325 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  unsigned short value = 0;

  //
  // Input argument descriptions for user help
  //
  std::string programDesc = "This simple program takes an unsigned short label \
map value that conforms to the labeling conventions laid out in \
cipConventhions.h and writes to the command line the corresponding chest \
region and chest type";

  std::string valueDesc = "The unsigned short label map value";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision: 325 $" );

    TCLAP::ValueArg<unsigned short> valueArg ( "v", "", valueDesc, true, value, "unsigned short", cl );

    cl.parse( argc, argv );

    value = valueArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  ChestConventions conventions;

  std::cout << "Chest Region:\t" << conventions.GetChestRegionNameFromValue( value ) << std::endl;
  std::cout << "Chest Value:\t"  << conventions.GetChestTypeNameFromValue( value ) << std::endl;

  return cip::EXITSUCCESS;
}

#endif
