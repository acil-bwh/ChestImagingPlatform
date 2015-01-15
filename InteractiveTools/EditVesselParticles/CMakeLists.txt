CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( EditVesselParticles )

SET( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" )

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )  
   INCLUDE ( ${VTK_USE_FILE} )    
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF( VTK_FOUND )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES(
${CMAKE_SOURCE_DIR}/Utilities
${CMAKE_SOURCE_DIR}/Common
)

SET ( EditVesselParticles_SOURCE EditVesselParticles.cxx )

IF ( APPLE )
   ADD_EXECUTABLE ( EditVesselParticles  MACOSX_BUNDLE ${EditVesselParticles_SOURCE})
ELSE ( APPLE )
   ADD_EXECUTABLE ( EditVesselParticles  ${EditVesselParticles_SOURCE})
ENDIF ( APPLE )
TARGET_LINK_LIBRARIES ( EditVesselParticles ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPUtilities CIPCommon )
