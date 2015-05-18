CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( EditAirwayParticles )

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
${CMAKE_SOURCE_DIR}/IO
)

SET ( EditAirwayParticles_SOURCE EditAirwayParticles.cxx )

IF ( APPLE )
   ADD_EXECUTABLE ( EditAirwayParticles  MACOSX_BUNDLE ${EditAirwayParticles_SOURCE})
ELSE ( APPLE )
   ADD_EXECUTABLE ( EditAirwayParticles  ${EditAirwayParticles_SOURCE})
ENDIF ( APPLE )

TARGET_LINK_LIBRARIES ( EditAirwayParticles ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPUtilities CIPCommon CIPIO )
