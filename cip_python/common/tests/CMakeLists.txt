# Include all the files in the baseline folder
FILE(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/baseline
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

ADD_TEST( NAME test_example COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_example.py )
ADD_TEST( NAME test_geometry_topology_data COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_geometry_topology_data.py )
