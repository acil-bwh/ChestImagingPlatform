CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateImageSubVolumes )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateImageSubVolumes_SOURCE GenerateImageSubVolumes.cxx )

ADD_EXECUTABLE ( GenerateImageSubVolumes ${GenerateImageSubVolumes_SOURCE})
TARGET_LINK_LIBRARIES ( GenerateImageSubVolumes ${ITK_LIBRARIES} )
