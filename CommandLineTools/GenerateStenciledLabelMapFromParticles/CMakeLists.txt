CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateStenciledLabelMapFromParticles )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateStenciledLabelMapFromParticles_SOURCE GenerateStenciledLabelMapFromParticles.cxx )

ADD_EXECUTABLE ( GenerateStenciledLabelMapFromParticles ${GenerateStenciledLabelMapFromParticles_SOURCE} )
TARGET_LINK_LIBRARIES ( GenerateStenciledLabelMapFromParticles ${VTK_LIBRARIES} ${ITK_LIBRARIES} CIPCommon )
