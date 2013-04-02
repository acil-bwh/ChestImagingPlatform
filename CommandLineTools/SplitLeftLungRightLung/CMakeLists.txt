CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( SplitLeftLungRightLung )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( SplitLeftLungRightLung_SOURCE SplitLeftLungRightLung.cxx )

ADD_EXECUTABLE ( SplitLeftLungRightLung ${SplitLeftLungRightLung_SOURCE})
TARGET_LINK_LIBRARIES ( SplitLeftLungRightLung ${ITK_LIBRARIES} )
