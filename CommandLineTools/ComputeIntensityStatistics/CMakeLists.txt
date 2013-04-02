CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ComputeIntensityStatistics )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( ComputeIntensityStatistics_SOURCE ComputeIntensityStatistics.cxx )

ADD_EXECUTABLE ( ComputeIntensityStatistics ${ComputeIntensityStatistics_SOURCE} )
TARGET_LINK_LIBRARIES ( ComputeIntensityStatistics ${ITK_LIBRARIES} )
