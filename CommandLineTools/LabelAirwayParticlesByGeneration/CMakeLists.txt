CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LabelAirwayParticlesByGeneration )

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build LabelAirwayParticlesByGeneration" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities/ITK )

SET ( LabelAirwayParticlesByGeneration_SOURCE LabelAirwayParticlesByGeneration.cxx )

ADD_EXECUTABLE ( LabelAirwayParticlesByGeneration ${LabelAirwayParticlesByGeneration_SOURCE}  )
TARGET_LINK_LIBRARIES ( LabelAirwayParticlesByGeneration ${VTK_LIBRARIES} ${ITK_LIBRARIES} CIPCommon )



