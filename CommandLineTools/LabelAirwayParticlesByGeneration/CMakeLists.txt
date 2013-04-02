CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LabelAirwayParticlesByGeneration )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build LabelAirwayParticlesByGeneration" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities/ITK )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

SET ( LabelAirwayParticlesByGeneration_SOURCE LabelAirwayParticlesByGeneration.cxx )

ADD_EXECUTABLE ( LabelAirwayParticlesByGeneration ${LabelAirwayParticlesByGeneration_SOURCE}  )
TARGET_LINK_LIBRARIES ( LabelAirwayParticlesByGeneration ${VTK_LIBRARIES} ${ITK_LIBRARIES} CIPCommon )



