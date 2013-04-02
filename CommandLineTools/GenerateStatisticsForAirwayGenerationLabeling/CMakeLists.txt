CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateStatisticsForAirwayGenerationLabeling )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build GenerateStatisticsForAirwayGenerationLabeling" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

SET ( GenerateStatisticsForAirwayGenerationLabeling_SOURCE GenerateStatisticsForAirwayGenerationLabeling.cxx )

ADD_EXECUTABLE ( GenerateStatisticsForAirwayGenerationLabeling ${GenerateStatisticsForAirwayGenerationLabeling_SOURCE}  )
TARGET_LINK_LIBRARIES ( GenerateStatisticsForAirwayGenerationLabeling ${VTK_LIBRARIES} CIPCommon )



