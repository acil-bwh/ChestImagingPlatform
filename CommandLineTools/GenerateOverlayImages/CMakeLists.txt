CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(GenerateOverlayImages)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( GenerateOverlayImages_SOURCE GenerateOverlayImages.cxx )

ADD_EXECUTABLE ( GenerateOverlayImages ${GenerateOverlayImages_SOURCE})
TARGET_LINK_LIBRARIES ( GenerateOverlayImages CIPCommon ${ITK_LIBRARIES} )



