ADD_TEST( NAME test_region_type_parser COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/utils/tests/test_region_type_parser.py ) 

ADD_TEST( NAME test_remap_lm COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/utils/tests/test_remap_lm.py ) 

ADD_TEST( NAME test_anonymize_dicom COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/utils/tests/test_anonymize_dicom.py ) 

ADD_TEST( NAME test_compute_dice_coefficient COMMAND nosetests ${CMAKE_SOURCE_DIR}/cip_python/utils/tests/test_compute_dice_coefficient.py ) 
