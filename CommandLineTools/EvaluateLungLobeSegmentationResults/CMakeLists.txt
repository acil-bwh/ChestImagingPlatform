CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( EvaluateLungLobeSegmentationResults )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( EvaluateLungLobeSegmentationResults_SOURCE EvaluateLungLobeSegmentationResults.cxx )

ADD_EXECUTABLE ( EvaluateLungLobeSegmentationResults ${EvaluateLungLobeSegmentationResults_SOURCE} )
TARGET_LINK_LIBRARIES ( EvaluateLungLobeSegmentationResults ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon CIPIO )
