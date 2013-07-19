CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RemoveChestTypeFromLabelMapUsingParticles )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( RemoveChestTypeFromLabelMapUsingParticles_SOURCE RemoveChestTypeFromLabelMapUsingParticles.cxx )

ADD_EXECUTABLE ( RemoveChestTypeFromLabelMapUsingParticles ${RemoveChestTypeFromLabelMapUsingParticles_SOURCE} )
TARGET_LINK_LIBRARIES ( RemoveChestTypeFromLabelMapUsingParticles ${VTK_LIBRARIES} ${ITK_LIBRARIES} )
