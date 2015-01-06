CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateImageSubVolumes )

SET ( MODULE_NAME GenerateImageSubVolumes )
SET ( MODULE_SRCS GenerateImageSubVolumes.cxx )

SET ( MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
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
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64_0000.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64_0000.nhdr
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64_0001.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64_0001.nhdr
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64_0002.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64_0002.nhdr
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64_0003.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64_0003.nhdr
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_lm-64_0000.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm-64_0000.nhdr
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_lm-64_0001.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm-64_0001.nhdr
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_lm-64_0002.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm-64_0002.nhdr
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_lm-64_0003.nhdr
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm-64_0003.nhdr
    ModuleEntryPoint
      -c ${INPUT_DATA_DIR}/ct-64.nrrd
      -l ${INPUT_DATA_DIR}/lm-64.nrrd
      --ctPre ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64_
      --lmPre ${OUTPUT_DATA_DIR}/${TEST_NAME}_lm-64_
      -o 1
      -r 50
      --wls
)