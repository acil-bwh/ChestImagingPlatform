cmake_minimum_required(VERSION 2.8.9)

PROJECT( CIP)


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
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

#---------------------------------------------------------------------
# Find VTK.

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

#---------------------------------------------------------------------
# Find Teem

FIND_PACKAGE ( Teem )
IF ( Teem_FOUND )
  INCLUDE(${Teem_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( Teem_FOUND )

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

set( LIBRARY_OUTPUT_PATH ${CIP_BINARY_DIR}/bin
  CACHE INTERNAL "Single output directory for building all libraries." )

set( EXECUTABLE_OUTPUT_PATH ${CIP_BINARY_DIR}/bin
  CACHE INTERNAL "Single output directory for building all executables." )

mark_as_advanced( LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH )

set( CIP_LIBRARY_PATH    "${LIBRARY_OUTPUT_PATH}" )
set( CIP_EXECUTABLE_PATH "${EXECUTABLE_OUTPUT_PATH}" )

#---------------------------------------------------------------------
# Include directories

set( CIP_INCLUDE_DIRECTORIES 
  "${CIP_SOURCE_DIR}/Common"
  "${CIP_SOURCE_DIR}/Utilities/ITK"
  "${CIP_SOURCE_DIR}/Utilities/VTK"
  "${CMAKE_BINARY_DIR}/Common"
  "${CMAKE_BINARY_DIR}/Utilities/VTK"
  "${CMAKE_BINARY_DIR}/Utilities/ITK"
)

include_directories( ${CIP_INCLUDE_DIRECTORIES} )

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
# Testing

set( CIP_BUILD_TESTING OFF CACHE BOOL
  "Perform some tests on basic functionality of CIP." )

if ( CIP_BUILD_TESTING )
  enable_testing()
  #add_subdirectory( ${CIP_SOURCE_DIR}/Testing ${CIP_BINARY_DIR}/Testing )
  include( CTest )
endif()

#---------------------------------------------------------------------
# Compilation options

SET(BUILD_UTILITIES ON CACHE BOOL "BUILD_UTILITIES")
IF(BUILD_UTILITIES)
  SUBDIRS (Utilities)
ENDIF(BUILD_UTILITIES)

SET(BUILD_COMMON ON CACHE BOOL "BUILD_COMMON")
IF(BUILD_COMMON)
  SUBDIRS (Common)
ENDIF(BUILD_COMMON)

SET(BUILD_IO ON CACHE BOOL "BUILD_IO")
IF(BUILD_IO)
  SUBDIRS (IO)
ENDIF(BUILD_IO)

SET(BUILD_COMMANDLINETOOLS ON CACHE BOOL "BUILD_COMMANDLINETOOLS")
IF(BUILD_COMMANDLINETOOLS)
  SUBDIRS (CommandLineTools)
ENDIF(BUILD_COMMANDLINETOOLS)

SET(BUILD_INTERACTIVETOOLS OFF CACHE BOOL "BUILD_INTERACTIVETOOLS")
IF(BUILD_INTERACTIVETOOLS)
  SUBDIRS (InteractiveTools)
ENDIF(BUILD_INTERACTIVETOOLS)

SET(BUILD_SANDBOX OFF CACHE BOOL "BUILD_SANDBOX")
IF(BUILD_SANDBOX)
  SUBDIRS (Sandbox)
ENDIF(BUILD_SANDBOX)

IF ( CIP_BUILD_TESTING )
 SUBDIRS ( cip_python )
ENDIF( CIP_BUILD_TESTING ) 

#----------------------------------------------------------------------
# Make it easier to include cip functionality in other programs.
# See UseFile.cmake for instructions.

# Save library dependencies. (NOT WORKING YET)
#SET( CIP_LIBRARY_DEPENDS_FILE ${CIP_BINARY_DIR}/CIPLibraries.cmake )
#EXPORT( TARGETS ${CIP_LIBRARIES} FILE ${CIP_LIBRARY_DEPENDS_FILE} )

# The "use" file.
SET( CIP_USE_FILE ${CIP_SOURCE_DIR}/UseFile.cmake )

# Create the config file. It defines some variables, and by placing
# this in the binary directory, the find_package command will recognise
# the CIP package.

CONFIGURE_FILE( ${CIP_SOURCE_DIR}/CIPConfig.cmake.in
 ${CIP_BINARY_DIR}/CIPConfig.cmake @ONLY )

# Store the current build directory in the CMake user package registry
# for package <name>. The find_package command may consider the director
# while searching for package <name>.

export( PACKAGE CIP )

