#include "cipLobeSurfaceModelIO.h"
#include "cipExceptionObject.h"

int main()
{
  int testPass = 1; // Indicates failure

  // Give the reader a bogus file name. An exception should be thrown,
  // otherwise the test fails.
  cip::LobeSurfaceModelIO io;
    io.SetFileName( "foo" );
  try
    {
      io.Read();
    }
  catch ( cip::ExceptionObject &excp )
    {
      // If we're here, the exception was correctly thrown and caught,
      // so pass the test
      testPass = 0; // Indicates success
    }

  return testPass;
}
