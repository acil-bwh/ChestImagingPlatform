#
# To use CIP-code in your own program, use the following
# cmake code:
#
# cmake_minimum_required(VERSION 2.8)
#
# # You may omit this if you use the ADD_CIPTOOL macro:
# project( CIPexternalprog )
#
# FIND_PACKAGE( ITK REQUIRED )
# IF( EXISTS ${ITK_USE_FILE} )
# include( ${ITK_USE_FILE} )
# ENDIF()
#
# FIND_PACKAGE( CIP REQUIRED )
# IF( EXISTS ${CIP_USE_FILE} )
#  INCLUDE( ${CIP_USE_FILE} )
# ENDIF()
#
# # Optionally, you could use the ADD_CIPTOOL macro:
# # Then your program yourownprogram.cxx will be automatically
# # linked to the correct libraries in most cases, and it will be
# # installed in the same directory as the other pxtools.
# # Make sure to have one file yourownprogram.cxx in the 
# # current directory, which defines a main() function. All other
# # files in the current directory are also added as source code for 
# # this executable.
# ADD_CIPTOOL( yourownprogram )
#
# # # Or do something like this:
# # ADD_EXECUTABLE( yourownprogram yourownprogram.cxx )
# # TARGET_LINK_LIBRARIES( yourownprogram
# #  ${CIP_LIBRARIES} ${ITK_LIBRARIES} )
# 

MESSAGE( STATUS "Including CIP settings.")

#---------------------------------------------------------------------
# Set all necessary include directories for CIP
INCLUDE_DIRECTORIES( ${CIP_INCLUDE_DIRECTORIES} )

# Set the directory that contains the CIP libraries,
LINK_DIRECTORIES( "${CIP_LIBRARY_OUTPUT_PATH}" )
LINK_DIRECTORIES( "${CIP_LIBRARY_PATH}" )

# Read in the library dependencies
INCLUDE( "${CIP_LIBRARY_DEPENDS_FILE}" )

#---------------------------------------------------------------------
# Increases address capacity
if ( WIN32 )
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj" )
  set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj" )
endif()

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
# Include the macro definitions
# to use file:
INCLUDE_DIRECTORIES( ${CIP_MACRO_DEFINITIONS} )


