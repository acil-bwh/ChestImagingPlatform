/** \file
 *  \ingroup commandLineTools 
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "ConvertChestRegionChestTypeToLabelMapValueCLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
    
  cip::ChestConventions conventions;
  unsigned char cipRegion = conventions.GetChestRegionValueFromName( region );
  unsigned char cipType = conventions.GetChestTypeValueFromName( type );
  std::cout<<"Value:\t" << conventions.GetValueFromChestRegionAndType( cipRegion, cipType ) << std::endl;

  return cip::EXITSUCCESS;
}

#endif
