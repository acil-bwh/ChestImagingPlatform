# Include all the files in the baseline
FILE(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/baseline
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

ADD_TEST( NAME test_grid_segmenter COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_grid_segmenter.py )
ADD_TEST( NAME test_lung_segmentation_dcnn COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_lung_segmentation_dcnn.py )
