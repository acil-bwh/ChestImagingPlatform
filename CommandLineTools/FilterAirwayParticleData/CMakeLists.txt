CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterAirwayParticleData )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( FilterAirwayParticleData_SOURCE FilterAirwayParticleData.cxx )

ADD_EXECUTABLE ( FilterAirwayParticleData ${FilterAirwayParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FilterAirwayParticleData ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon )
