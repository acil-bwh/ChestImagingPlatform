CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(GenerateInteractiveLobeSegmentationFissureMask )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateInteractiveLobeSegmentationFissureMask_SOURCE GenerateInteractiveLobeSegmentationFissureMask.cxx )

ADD_EXECUTABLE ( GenerateInteractiveLobeSegmentationFissureMask ${GenerateInteractiveLobeSegmentationFissureMask_SOURCE} )
TARGET_LINK_LIBRARIES ( GenerateInteractiveLobeSegmentationFissureMask ${ITK_LIBRARIES} )
