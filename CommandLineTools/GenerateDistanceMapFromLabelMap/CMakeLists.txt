CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateDistanceMapFromLabelMap )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateDistanceMapFromLabelMap_SOURCE GenerateDistanceMapFromLabelMap.cxx )

ADD_EXECUTABLE ( GenerateDistanceMapFromLabelMap ${GenerateDistanceMapFromLabelMap_SOURCE} )
TARGET_LINK_LIBRARIES ( GenerateDistanceMapFromLabelMap ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon)
