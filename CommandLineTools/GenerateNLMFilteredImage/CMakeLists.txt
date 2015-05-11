cmake_minimum_required(VERSION 2.6.0)

PROJECT ( GenerateNLMFilteredImage )

SET ( MODULE_NAME GenerateNLMFilteredImage )
SET( MODULE_SRCS GenerateNLMFilteredImage.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/ct-64.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
      -s 10 
      --rs 11,11,0
)