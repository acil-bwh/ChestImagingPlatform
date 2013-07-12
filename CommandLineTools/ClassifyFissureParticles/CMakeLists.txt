CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ClassifyFissureParticles )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( ClassifyFissureParticles_SOURCE ClassifyFissureParticles.cxx )

ADD_EXECUTABLE ( ClassifyFissureParticles ${ClassifyFissureParticles_SOURCE})
TARGET_LINK_LIBRARIES ( ClassifyFissureParticles ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPIO CIPCommon  )
