CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateStatisticsForAirwayGenerationLabeling )

set(MODULE_NAME GenerateStatisticsForAirwayGenerationLabeling)

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build GenerateStatisticsForAirwayGenerationLabeling" )
ENDIF(NOT BUILD_COMMON)

set(MODULE_SRCS
  )
  
set(MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )
    



