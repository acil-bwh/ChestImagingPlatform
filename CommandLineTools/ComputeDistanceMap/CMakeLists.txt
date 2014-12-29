CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ComputeDistanceMap )

SET ( MODULE_NAME ComputeDistanceMap )
SET ( MODULE_SRCS ComputeDistanceMap.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )
    
SET (TEST_NAME ${MODULE_NAME}_Test)
add_test(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_dm.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dm.nrrd
    ModuleEntryPoint
      -l ${INPUT_DATA_DIR}/wholelung-64.nrrd
      -d ${OUTPUT_DATA_DIR}/${TEST_NAME}_dm.nrrd
)