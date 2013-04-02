CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ReadWriteRegionAndTypePoints )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( ReadWriteRegionAndTypePoints_SOURCE ReadWriteRegionAndTypePoints.cxx )

ADD_EXECUTABLE ( ReadWriteRegionAndTypePoints ${ReadWriteRegionAndTypePoints_SOURCE} )
TARGET_LINK_LIBRARIES ( ReadWriteRegionAndTypePoints CIPIO)
