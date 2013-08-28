CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GeneratePartialLungLabelMap )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( PerformMorphological_SOURCE GeneratePartialLungLabelMap.cxx )

ADD_EXECUTABLE ( GeneratePartialLungLabelMap ${PerformMorphological_SOURCE} )
TARGET_LINK_LIBRARIES ( GeneratePartialLungLabelMap ${ITK_LIBRARIES}  CIPCommon)
