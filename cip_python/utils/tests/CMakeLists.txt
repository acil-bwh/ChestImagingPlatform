ADD_TEST( NAME test_anonymize_dicom COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_anonymize_dicom.py )

ADD_TEST( NAME test_compute_dice_coefficient COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_compute_dice_coefficient.py )

ADD_TEST( NAME test_feature_maps COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_feature_maps.py )

ADD_TEST( NAME test_read_geometrytopologypoint_write_dataframe COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_read_geometrytopologypoint_write_dataframe.py )

ADD_TEST( NAME test_read_geometrytopologypoint_write_polydata COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_read_geometrytopologypoint_write_polydata.py )

ADD_TEST( NAME test_region_type_parser COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_region_type_parser.py )

ADD_TEST( NAME test_remap_lm COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_remap_lm.py )

ADD_TEST( NAME test_weighted_feature_map_densities COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_weighted_feature_map_densities.py )






