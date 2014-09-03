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

ADD_TEST( cipLobeSurfaceModelIOTEST cipLobeSurfaceModelIOTEST )
