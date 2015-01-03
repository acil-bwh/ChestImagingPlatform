CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ExtractChestLabelMap )

SET ( MODULE_NAME ExtractChestLabelMap )
SET ( MODULE_SRCS ExtractChestLabelMap.cxx )

SET ( MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test_RightLung)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
      -r RightLung
)

SET (TEST_NAME ${MODULE_NAME}_Test_LeftLung)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
      -r LeftLung
)

SET (TEST_NAME ${MODULE_NAME}_Test_WholeLung)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
      -r WholeLung
)

SET (TEST_NAME ${MODULE_NAME}_Test_Vessel)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
      -t Vessel
)

SET (TEST_NAME ${MODULE_NAME}_Test_RightLung_Vessel)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.nrrd
      -b RightLung
      -w Vessel
)
