CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateStatisticsForAirwayGenerationLabeling )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build GenerateStatisticsForAirwayGenerationLabeling" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateStatisticsForAirwayGenerationLabeling_SOURCE GenerateStatisticsForAirwayGenerationLabeling.cxx )

ADD_EXECUTABLE ( GenerateStatisticsForAirwayGenerationLabeling ${GenerateStatisticsForAirwayGenerationLabeling_SOURCE}  )
TARGET_LINK_LIBRARIES ( GenerateStatisticsForAirwayGenerationLabeling ${VTK_LIBRARIES} CIPCommon )



