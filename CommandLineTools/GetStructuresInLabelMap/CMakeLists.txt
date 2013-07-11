CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GetStructuresInLabelMap )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GetStructuresInLabelMap_SOURCE GetStructuresInLabelMap.cxx )

ADD_EXECUTABLE ( GetStructuresInLabelMap ${GetStructuresInLabelMap_SOURCE} )
TARGET_LINK_LIBRARIES ( GetStructuresInLabelMap ${ITK_LIBRARIES}  )
