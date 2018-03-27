#-----------------------------------
# cipChestRegionChestTypeLocationsIOTEST
#-----------------------------------
ADD_EXECUTABLE( cipChestRegionChestTypeLocationsIOTEST cipChestRegionChestTypeLocationsIOTEST.cxx)
TARGET_LINK_LIBRARIES( cipChestRegionChestTypeLocationsIOTEST CIPIO )

SET_TARGET_PROPERTIES ( cipChestRegionChestTypeLocationsIOTEST
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/IO/Testing"
)

ADD_TEST( cipChestRegionChestTypeLocationsIOTEST cipChestRegionChestTypeLocationsIOTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/Case000_regionAndTypePoints.csv )

#-----------------------------------
# cipLobeSurfaceModelIOTEST
#-----------------------------------
PROJECT ( cipLobeSurfaceModelIOTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

ADD_EXECUTABLE( cipLobeSurfaceModelIOTEST cipLobeSurfaceModelIOTEST.cxx)
TARGET_LINK_LIBRARIES( cipLobeSurfaceModelIOTEST CIPIO )

SET_TARGET_PROPERTIES ( cipLobeSurfaceModelIOTEST
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/IO/Testing"
)

ADD_TEST( cipLobeSurfaceModelIOTEST cipLobeSurfaceModelIOTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/Case000_leftLungLobesShapeModel.csv )

#-----------------------------------
# cipGeometryTopologyDataIOTEST
#-----------------------------------
PROJECT ( cipGeometryTopologyDataIOTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

ADD_EXECUTABLE( cipGeometryTopologyDataIOTEST cipGeometryTopologyDataIOTEST.cxx)
TARGET_LINK_LIBRARIES( cipGeometryTopologyDataIOTEST CIPIO )

SET_TARGET_PROPERTIES ( cipGeometryTopologyDataIOTEST
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/IO/Testing"
)

ADD_TEST( cipGeometryTopologyDataIOTEST cipGeometryTopologyDataIOTEST
	  ${CMAKE_SOURCE_DIR}/Testing/Data/Input/geometryTopologyData.xml ${CIP_BINARY_DIR}/IO/Testing/cipGeometryTopologyDataIOTEST_out.xml)
