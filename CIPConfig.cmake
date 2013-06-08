#
# See UseCIP.cmake for explanation how to use CIP code
# from an external project.
#

MESSAGE( STATUS "Configuring CIP" )

#---------------------------------------------------------------------
# Set all necessary include directories for CIP
SET( CIP_INCLUDE_DIRECTORIES "/projects/lmi/people/rharmo/projects/ChestImagingPlatformPrivate/Common;/projects/lmi/people/rharmo/projects/ChestImagingPlatformPrivate/Utilities/ITK;/projects/lmi/people/rharmo/projects/ChestImagingPlatformPrivate/Utilities/VTK" )

# Set the directory that contains the CIP libraries,
# such as CIP-Common.
SET( CIP_LIBRARY_OUTPUT_PATH "/projects/lmi/people/rharmo/projects/ChestImagingPlatformPrivate/bin" )

# Read in the library dependencies
SET( CIP_LIBRARY_DEPENDS_FILE "" )

#---------------------------------------------------------------------
# Set some variables that the user might want to use
SET( CIP_LIBRARIES CIPCommon;CIPUtilities )
SET( CIP_INSTALL_DIR "/usr/local/bin" )
SET( CIP_MACRO_DEFINITIONS "" )
SET( CIP_USE_FILE "/projects/lmi/people/rharmo/projects/ChestImagingPlatformPrivate/UseFile.cmake" )

