CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterVesselParticleData )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( FilterVesselParticleData_SOURCE FilterVesselParticleData.cxx )

ADD_EXECUTABLE ( FilterVesselParticleData ${FilterVesselParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FilterVesselParticleData ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPCommon )
