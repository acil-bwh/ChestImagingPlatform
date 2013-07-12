CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FitFissureSurfaceModelToParticleData )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( FitFissureSurfaceModelToParticleData_SOURCE FitFissureSurfaceModelToParticleData.cxx )

ADD_EXECUTABLE ( FitFissureSurfaceModelToParticleData ${FitFissureSurfaceModelToParticleData_SOURCE} )
TARGET_LINK_LIBRARIES ( FitFissureSurfaceModelToParticleData ${ITK_LIBRARIES}  ${VTK_LIBRARIES}  CIPIO CIPCommon )
