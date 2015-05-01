set(TEST_NAME vtkSmoothLines)
project (Test_${TEST_NAME})
add_executable(Test_${TEST_NAME} test.cxx)
target_link_libraries(Test_${TEST_NAME} ${CIP_LIBRARIES})
