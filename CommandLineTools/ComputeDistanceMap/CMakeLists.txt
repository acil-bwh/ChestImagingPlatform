CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ComputeDistanceMap )


FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( ComputeDistanceMap_SOURCE ComputeDistanceMap.cxx )

ADD_EXECUTABLE ( ComputeDistanceMap ${ComputeDistanceMap_SOURCE} )
TARGET_LINK_LIBRARIES ( ComputeDistanceMap ${ITK_LIBRARIES} CIPCommon)
