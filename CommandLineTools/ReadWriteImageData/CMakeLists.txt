CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ReadWriteImageData )

set(MODULE_NAME ReadWriteImageData)

set(MODULE_SRCS
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )
    
SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_simple_ct.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_ct.nrrd
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_simple_lm.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_lm.nrrd
    ModuleEntryPoint
      --ict ${INPUT_DATA_DIR}/simple_ct.nrrd
      --il ${INPUT_DATA_DIR}/simple_lm.nrrd
      --oct ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_ct.nrrd
      --ol ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_lm.nrrd
)