#include "itkTestMain.h"

#if defined(WIN32) && !defined(USE_STATIC_CIP_LIBS)
#define MODULE_IMPORT __declspec(dllimport)
#else
#define MODULE_IMPORT
#endif

// Comment copied from ThesholdTest.cxx; This will be linked against the ModuleEntryPoint in RealignLib
extern "C" MODULE_IMPORT int ModuleEntryPoint(int, char *[]);


void RegisterTests()
{
	StringToTestFunctionMap["ModuleEntryPoint"] = ModuleEntryPoint;
}