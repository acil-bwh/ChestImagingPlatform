CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RelabelPecLabelMaps  )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( RelabelPecLabelMaps_SOURCE RelabelPecLabelMaps.cxx )

ADD_EXECUTABLE ( RelabelPecLabelMaps ${RelabelPecLabelMaps_SOURCE} )
TARGET_LINK_LIBRARIES ( RelabelPecLabelMaps ${ITK_LIBRARIES}  )
