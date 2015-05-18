CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RemapLabelMap )

set(MODULE_NAME RemapLabelMap )

set(MODULE_SRCS )
  
cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMapDice  
      ${BASELINE_DATA_DIR}/${TEST_NAME}_lm.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm.nrrd
    --diceTolerance 100
    ModuleEntryPoint
      --ilm ${INPUT_DATA_DIR}/simple_lm.nrrd
      --olm ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm.nrrd
      --il 515,2,771 
      --ol 15,2,3
)

