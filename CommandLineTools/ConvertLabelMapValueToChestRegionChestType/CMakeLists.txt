CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ConvertLabelMapValueToChestRegionChestType )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( ConvertLabelMapValueToChestRegionChestType_SOURCE ConvertLabelMapValueToChestRegionChestType.cxx )

ADD_EXECUTABLE ( ConvertLabelMapValueToChestRegionChestType ${ConvertLabelMapValueToChestRegionChestType_SOURCE} )
TARGET_LINK_LIBRARIES ( ConvertLabelMapValueToChestRegionChestType ${ITK_LIBRARIES}  )
