CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ExtractParticlesFromChestRegionChestType )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( ExtractParticlesFromChestRegionChestType_SOURCE ExtractParticlesFromChestRegionChestType.cxx )

ADD_EXECUTABLE ( ExtractParticlesFromChestRegionChestType ${ExtractParticlesFromChestRegionChestType_SOURCE})
TARGET_LINK_LIBRARIES ( ExtractParticlesFromChestRegionChestType ${ITK_LIBRARIES} ${VTK_LIBRARIES} )
