# Include all the files in the baseline
FILE(INSTALL ${SOURCE_DIR}/baseline
    DESTINATION ${TARGET_DIR}/${PYTHON_MODULE}/baseline
    FILES_MATCHING PATTERN "*.*"
)

ADD_TEST( NAME test_grid_segmenter COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_grid_segmenter.py )
ADD_TEST( NAME test_lung_segmentation_dcnn COMMAND nosetests ${CMAKE_CURRENT_BINARY_DIR}/test_lung_segmentation_dcnn.py )
