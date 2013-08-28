CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FitFissureSurfaceModelsToParticleData )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( FitFissureSurfaceModelsToParticleData_SOURCE FitFissureSurfaceModelsToParticleData.cxx )

ADD_EXECUTABLE ( FitFissureSurfaceModelsToParticleData ${FitFissureSurfaceModelsToParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FitFissureSurfaceModelsToParticleData ${ITK_LIBRARIES} ${VTK_LIBRARIES}  CIPIO CIPCommon )
