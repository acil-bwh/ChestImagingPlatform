CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterFissureParticleData )

SET( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( FilterFissureParticleData_SOURCE FilterFissureParticleData.cxx )

ADD_EXECUTABLE ( FilterFissureParticleData ${FilterFissureParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FilterFissureParticleData ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon )

