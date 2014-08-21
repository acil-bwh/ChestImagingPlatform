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

#include "cipChestConventions.h"
#include "ConvertChestRegionChestTypeToLabelMapValueCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
    
  cip::ChestConventions conventions;
  std::cout<<"Value:\t" << conventions.GetValueFromChestRegionAndType((unsigned char)region,(unsigned char)type) << std::endl;

  return cip::EXITSUCCESS;
}

#endif
