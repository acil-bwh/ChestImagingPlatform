cmake_minimum_required(VERSION 3.5.0)

##################### RPath stuff start ################
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")
############## RPATH Stuff end #########################

#-----------------------------------------------------------------------------
# Update CMake module path
#------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
  ${CIP_SOURCE_DIR}/CMake
  ${CMAKE_MODULE_PATH}
  )
set(CIP_CMAKE_DIR ${CIP_SOURCE_DIR}/CMake)


#--------------------------------------------------------------------
# Find ITK.
FIND_PACKAGE ( ITK )
if ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
else ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
endif ( ITK_FOUND )

#---------------------------------------------------------------------
# Find VTK.

FIND_PACKAGE ( VTK REQUIRED NO_MODULE)
if ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
else ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
endif ( VTK_FOUND )

#Check Boost status
if(TARGET vtkInfovisBoostGraphAlgorithms)   # Make sure we have the needed module
  message(STATUS "Using Infovis Boost")
  SET(CIP_USE_BOOST ON)
else()
  message(STATUS "vtkInfovisBoostGraphAlgorithms not found! CIP will NOT use Boost")
  SET(CIP_USE_BOOST OFF)
endif()




  #---------------------------------------------------------------------
# Find Teem

FIND_PACKAGE ( Teem )
if ( Teem_FOUND )
  INCLUDE(${Teem_USE_FILE})
else ( Teem_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without Teem" )
endif( Teem_FOUND )


#---------------------------------------------------------------------
# Find OpenCV

#FIND_PACKAGE ( OpenCV )

#---------------------------------------------------------------------
# Kill the anoying MS VS warning about non-safe functions.
# They hide real warnings.

if( MSVC )
add_definitions(
    /D_SCL_SECURE_NO_DEPRECATE
    /D_CRT_SECURE_NO_DEPRECATE
    /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
    )
endif()

#---------------------------------------------------------------------
# Increases address capacity

if( WIN32 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" )
endif()

#---------------------------------------------------------------------
# Output directories.

if( CIP_INTEGRATE_WITH_SLICER ) # may be set with remote module options

  set( LIBRARY_OUTPUT_PATH ${CIP_CLI_LIBRARY_OUTPUT_DIRECTORY} )
  set( EXECUTABLE_OUTPUT_PATH ${CIP_CLI_RUNTIME_OUTPUT_DIRECTORY} )

else()
  
  set( LIBRARY_OUTPUT_PATH ${CIP_BINARY_DIR}/bin
    CACHE INTERNAL "Single output directory for building all libraries." )

  set( EXECUTABLE_OUTPUT_PATH ${CIP_BINARY_DIR}/bin
    CACHE INTERNAL "Single output directory for building all executables." )

endif()

mark_as_advanced( LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH )

get_filename_component( CIP_PARENT_DIR ${CMAKE_BINARY_DIR} PATH )

set( CIP_LIBRARY_PATH "${CIP_PARENT_DIR}/lib" )
set( CIP_EXECUTABLE_PATH "${EXECUTABLE_OUTPUT_PATH}" )
get_filename_component(CIP_PYTHON_BIN_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
if (WIN32)
  set(CIP_PYTHON_BIN_DIR ${CIP_PYTHON_BIN_DIR}/Scripts)
endif()
#---------------------------------------------------------------------
# Testing

set( CIP_BUILD_TESTING ON CACHE BOOL "Perform some tests on basic functionality of CIP." )

if ( CIP_BUILD_TESTING )
  enable_testing()
  include( CTest )  
  SET(CIP_BUILD_TESTING_LARGE OFF CACHE BOOL "Build large tests that require MIDAS server")  
  SET(CIP_BUILD_TESTING_PYTHON ON CACHE BOOL "Build Python tests") 
  # Copy the Input data for testing to the build directory (same folder structure, so tests will work both in source and build)
  ADD_CUSTOM_TARGET(copy-input-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Testing/Data/Input ${CMAKE_CURRENT_BINARY_DIR}/Testing/Data/Input
  )
  if (CIP_BUILD_TESTING_PYTHON)
    # Set the nosetests executable that will be used for python tests
    set (CIP_NOSETESTS_EXEC "${CIP_PYTHON_BIN_DIR}/nosetests" CACHE FILEPATH "nosetests executable" FORCE)
  endif()
else( CIP_BUILD_TESTING )
  SET(CIP_BUILD_TESTING_LARGE OFF CACHE BOOL "Build large tests that require MIDAS server")
  SET(CIP_BUILD_TESTING_PYTHON OFF CACHE BOOL "Build Python tests")
endif( CIP_BUILD_TESTING )

#---------------------------------------------------------------------
# MIDAS configuration (used in LARGE Testing to store data files)
if ( CIP_BUILD_TESTING_LARGE )    
  include(MIDAS)
  include(MIDASAPILogin)
 
  set(MIDAS_REST_URL "http://midas.chestimagingplatform.org/rest" CACHE STRING "The MIDAS server where testing data resides")
  set(MIDAS_KEY_DIR "${CMAKE_SOURCE_DIR}/Testing/Data/Large/MIDAS_Keys" CACHE PATH "Directory where hash files are stored (included in Source code)")
  set(MIDAS_DATA_DIR "${CIP_BINARY_DIR}/Testing/Data/Large/MIDAS_Data" CACHE PATH "Directory where downloaded files are stored (local directory)")
  FILE(MAKE_DIRECTORY ${MIDAS_DATA_DIR})

  set(MIDAS_USER_EMAIL "" CACHE STRING "Email used to authenticate MIDAS server")
  set(MIDAS_USER_APIKEY "" CACHE STRING "API key of the user used to authenticate MIDAS server")
  

  # # Authenticate with MIDAS (Public user by default, but settings can be modified)
  if ((MIDAS_USER_EMAIL) AND (MIDAS_USER_APIKEY))
    midas_api_login(
          MIDAS_REST_URL ${MIDAS_REST_URL}
          MIDAS_USER_EMAIL ${MIDAS_USER_EMAIL}
          MIDAS_USER_APIKEY ${MIDAS_USER_APIKEY}
          RESULT_VARNAME MIDAS_AUTH_TOKEN
    )
  endif()

  if (NOT MIDAS_AUTH_TOKEN)
    MESSAGE("Warning: MIDAS authentication could not be completed. Please make sure that MIDAS_USER_EMAIL and MIDAS_USER_APIKEY settings are correct")
    SET (MIDAS_AUTH_TOKEN 0)    # Init with dumb value to avoid build errors
  endif()
endif()


#---------------------------------------------------------------------
# Include directories

set( CIP_INCLUDE_DIRECTORIES 
  "${CIP_SOURCE_DIR}/Common"
  "${CIP_SOURCE_DIR}/Utilities/ITK"
  "${CIP_SOURCE_DIR}/Utilities/VTK"
  "${CIP_SOURCE_DIR}/Utilities/LesionSizingToolkit"
  "${CIP_SOURCE_DIR}/Utilities/GraphCutsOptimization"
  "${CIP_SOURCE_DIR}/Utilities/ITKStrain"
  "${CIP_SOURCE_DIR}/IO"
  "${CIP_BINARY_DIR}/Common"
  "${CIP_BINARY_DIR}/Utilities/VTK"
  "${CIP_BINARY_DIR}/Utilities/ITK"
  "${CIP_BINARY_DIR}/Utilities/LesionSizingToolkit"
  "${CIP_BINARY_DIR}/Utilities/GraphCutsOptimization"
  "${CIP_BINARY_DIR}/Utilities/ITKStrain"
  "${CIP_BINARY_DIR}/IO"
)

include_directories( ${CIP_INCLUDE_DIRECTORIES} )
#---------------------------------------------------------------------
# Copy external files
ADD_CUSTOM_TARGET(copy-external-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Resources ${CMAKE_CURRENT_BINARY_DIR}/Resources
    COMMENT "Copying Resource files"
  )


#---------------------------------------------------------------------
# Link libraries

SET( CIP_LIBRARIES CIPCommon CIPUtilities CIPIO)

#---------------------------------------------------------------------
# Define where to install CIP

if( WIN32 )
set( CIP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
else()  
set( CIP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin )
endif()
 

#---------------------------------------------------------------------
set( CIP_BUILD_CLI_EXECUTABLEONLY ON CACHE BOOL "Build CLIs only with executables and not shared libraries+executables.")

#---------------------------------------------------------------------
# Compilation options

SET(BUILD_UTILITIES ON CACHE BOOL "BUILD_UTILITIES")
if(BUILD_UTILITIES)
  SUBDIRS (Utilities)
endif(BUILD_UTILITIES)

SET(BUILD_COMMON ON CACHE BOOL "BUILD_COMMON")
if(BUILD_COMMON)
  SUBDIRS (Common)
endif(BUILD_COMMON)

SET(BUILD_IO ON CACHE BOOL "BUILD_IO")
if(BUILD_IO)
  SUBDIRS (IO)
endif(BUILD_IO)

SET(BUILD_COMMANDLINETOOLS ON CACHE BOOL "BUILD_COMMANDLINETOOLS")
if(BUILD_COMMANDLINETOOLS)
  SUBDIRS (CommandLineTools)
endif(BUILD_COMMANDLINETOOLS)

SET(BUILD_INTERACTIVETOOLS OFF CACHE BOOL "BUILD_INTERACTIVETOOLS")
if(BUILD_INTERACTIVETOOLS)
  SUBDIRS (InteractiveTools)
endif(BUILD_INTERACTIVETOOLS)

SET(BUILD_SANDBOX OFF CACHE BOOL "BUILD_SANDBOX")
if(BUILD_SANDBOX)
  SUBDIRS (Sandbox)
endif(BUILD_SANDBOX)

SUBDIRS ( cip_python )
  # Option to disable GCO wrapping when cip python is not going to be needed.
SET(USE_CYTHON ON CACHE BOOL "Use of Cython, needed to Wrap GraphCutsOptimization for Python")
mark_as_advanced(FORCE USE_CYTHON)


#-----------------------------------------------------------------------------
# CMake Function(s) and Macro(s)
#-----------------------------------------------------------------------------
include(cipMacroBuildCLI)


#----------------------------------------------------------------------
# Make it easier to include cip functionality in other programs.
# See UseFile.cmake for instructions.

# Save library dependencies. (NOT WORKING YET)
#SET( CIP_LIBRARY_DEPENDS_FILE ${CIP_BINARY_DIR}/CIPLibraries.cmake )
#EXPORT( TARGETS ${CIP_LIBRARIES} FILE ${CIP_LIBRARY_DEPENDS_FILE} )

# Added to provide a way to find CIPConfig.cmake from internal sub-projects
# Usually this is achieved by export command (below) but it sometime fails
# to create package registry (under .cmake directory) for some reason...
SET( CIP_DIR ${CIP_BINARY_DIR} )



# The "use" file.
SET( CIP_USE_FILE ${CIP_CMAKE_DIR}/UseFile.cmake )

# Create the config file. It defines some variables, and by placing
# this in the binary directory, the find_package command will recognise
# the CIP package.

CONFIGURE_FILE( ${CIP_CMAKE_DIR}/CIPConfig.cmake.in
 ${CIP_BINARY_DIR}/CIPConfig.cmake @ONLY )

# Store the current build directory in the CMake user package registry
# for package <name>. The find_package command may consider the director
# while searching for package <name>.

#-----------------------------------------------------------------------------
# Avoid linker bug in Mac OS 10.5
# See http://wiki.finkproject.org/index.php/Fink:Packaging:Preparing_for_10.5#OpenGL_Bug
#
if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-dylib_file,/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,-dylib_file,/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib")
  #Due to the change to clang on OS X 1.9, you have to build against the old libs (libstdc++ and not the clang one - libc++).
  #set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -mmacosx-version-min=10.6")
  #set (CMAKE_CXX_FLAGS ${CIP_cxx_flags})
endif()

    #-----------------------------------------------------------------------------
# Add needed flag for gnu on linux like enviroments to build static common libs
# suitable for linking with shared object libs.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  if(NOT "${CMAKE_CXX_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
  if(NOT "${CMAKE_C_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

export( PACKAGE CIP )

