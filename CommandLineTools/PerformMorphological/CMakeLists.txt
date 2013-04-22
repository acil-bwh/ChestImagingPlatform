CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( PerformMorphological )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( PerformMorphological_SOURCE PerformMorphological.cxx )

ADD_EXECUTABLE ( PerformMorphological ${PerformMorphological_SOURCE} )
TARGET_LINK_LIBRARIES ( PerformMorphological ${ITK_LIBRARIES}  CIPCommon)
