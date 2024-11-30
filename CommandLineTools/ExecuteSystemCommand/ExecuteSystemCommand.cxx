#include <stdio.h>      
#include <stdlib.h>
#include <fstream>

#include "cipChestConventions.h"

#include "ExecuteSystemCommandCLP.h"



int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  const char* strCommand = command.c_str();
  std::cout << "** Executing the following command: " << strCommand << std::endl;
  int result = system(strCommand);
  std::cout << "Result of the command: " << result << std::endl;
  std::cout << "** END **" << std::endl;

  std::ofstream writeFile (returnParameterFile.c_str());

  writeFile << "output = " << result << std::endl;
  writeFile.close();

  std::cout << "Results saved in the file " << returnParameterFile.c_str() << std::endl;

  if (result == 0)
  	// OK case
  	return cip::EXITSUCCESS;
  // Something failed in the command
  return cip::EXITFAILURE;
}
