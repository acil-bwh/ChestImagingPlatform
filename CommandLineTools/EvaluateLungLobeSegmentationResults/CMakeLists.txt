CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( EvaluateLungLobeSegmentationResults )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )
  INCLUDE (${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF (VTK_FOUND)


INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( EvaluateLungLobeSegmentationResults_SOURCE EvaluateLungLobeSegmentationResults.cxx )

ADD_EXECUTABLE ( EvaluateLungLobeSegmentationResults ${EvaluateLungLobeSegmentationResults_SOURCE} )
TARGET_LINK_LIBRARIES ( EvaluateLungLobeSegmentationResults ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon CIPIO )
