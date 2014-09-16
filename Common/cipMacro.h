#ifndef __cipMacro_h
#define __cipMacro_h

#include <iostream>

#define cipAssert(x) \
  if ( !(x) )	     \
    { \
      std::cerr << std::endl << "CIP Assertion Error:" << std::endl;	\
      std::cerr << "In file:\t" << __FILE__ << std::endl;			\
      std::cerr << "On line:\t" << __LINE__ << std::endl;			\
      std::cerr << "Aborting!" << std::endl;				\
      abort();					   \
    } 

#endif
