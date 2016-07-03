CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CIPIO )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build IO" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common ${VTK_LIBXML_INCLUDE_DIR})

SET ( CIPIO_SRCS
  cipLobeSurfaceModelIO.cxx
  cipChestRegionChestTypeLocationsIO.cxx
  cipThinPlateSplineSurfaceIO.cxx
  cipGeometryTopologyDataIO.cxx
  )

set( LIB_NAME CIPIO )

add_library(${LIB_NAME}
  ${CIPIO_SRCS} 
  )

TARGET_LINK_LIBRARIES ( ${LIB_NAME} ${VTK_LIBRARIES} ${ITK_LIBRARIES} ${LIBXML2_LIBRARIES} CIPCommon )


# --------------------------------------------------------------------------
# Install library
# --------------------------------------------------------------------------

if(NOT DEFINED CIP_INSTALL_LIB_DIR)
  set(CIP_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${LIB_NAME}
  RUNTIME DESTINATION ${CIP_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
  LIBRARY DESTINATION ${CIP_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CIP_INSTALL_LIB_DIR} COMPONENT Development)

IF ( CIP_BUILD_TESTING )
 SUBDIRS ( Testing )
ENDIF( CIP_BUILD_TESTING ) 
