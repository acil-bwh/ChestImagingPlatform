CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CIPIO )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build IO" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )
  INCLUDE (${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF (VTK_FOUND)

FIND_PACKAGE( ITK )
IF ( ITK_FOUND )
  INCLUDE (${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF (ITK_FOUND)

SET ( CIPIO_SRCS
  cipLobeBoundaryShapeModelIO.cxx
  cipChestRegionChestTypeLocationsIO.cxx
  cipThinPlateSplineSurfaceIO.cxx
  )

set( LIB_NAME CIPIO )

add_library(${LIB_NAME}
  ${CIPIO_SRCS} 
  )

TARGET_LINK_LIBRARIES ( ${LIB_NAME} ${VTK_LIBRARIES} ${ITK_LIBRARIES} CIPCommon )
