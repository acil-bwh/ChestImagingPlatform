CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ComputeIntensityStatistics )

set(MODULE_NAME ComputeIntensityStatistics)
 
#
# SlicerExecutionModel
#
#find_package(SlicerExecutionModel REQUIRED)
#include(${SlicerExecutionModel_USE_FILE})


set(MODULE_SRCS)
  
cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCSV
      ${BASELINE_DATA_DIR}/${TEST_NAME}.csv
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.csv
    ModuleEntryPoint
      -c ${INPUT_DATA_DIR}/ct-64.nrrd
      -l ${INPUT_DATA_DIR}/lm-64.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.csv
)