CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ReadParticlesWriteConnectedParticles)

IF(NOT BUILD_COMMON)
  MESSAGE ( FATAL_ERROR "Must build Common to build LabelAirwayParticlesByGeneration" )
ENDIF(NOT BUILD_COMMON)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET (ReadParticlesWriteConnectedParticles_SOURCE ReadParticlesWriteConnectedParticles.cxx )

ADD_EXECUTABLE (ReadParticlesWriteConnectedParticles ${ReadParticlesWriteConnectedParticles_SOURCE})
TARGET_LINK_LIBRARIES (ReadParticlesWriteConnectedParticles ${VTK_LIBRARIES} CIPCommon)



