ADD_TEST( NAME test_grid_segmenter COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_grid_segmenter.py )

ADD_TEST( NAME test_rind_vs_core_partition COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_rind_vs_core_partition.py )

