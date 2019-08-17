# Include all the files in the baseline folder
FILE(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/baseline
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

ADD_TEST( NAME test_low_to_high_dose_filter_dcnn COMMAND ${CIP_NOSETESTS_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/test_low_to_high_dose_filter_dcnn.py )