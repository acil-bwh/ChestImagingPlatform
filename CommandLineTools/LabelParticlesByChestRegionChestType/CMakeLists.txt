CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LabelParticlesByChestRegionChestType )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( LabelParticlesByChestRegionChestType_SOURCE LabelParticlesByChestRegionChestType.cxx )

ADD_EXECUTABLE ( LabelParticlesByChestRegionChestType ${LabelParticlesByChestRegionChestType_SOURCE} )
TARGET_LINK_LIBRARIES ( LabelParticlesByChestRegionChestType CIPCommon ${VTK_LIBRARIES} ${ITK_LIBRARIES} )

