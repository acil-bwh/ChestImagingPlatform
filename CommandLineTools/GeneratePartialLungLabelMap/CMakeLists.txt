CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GeneratePartialLungLabelMap )

set(MODULE_NAME GeneratePartialLungLabelMap)

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
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
    ModuleEntryPoint
      --ict ${INPUT_DATA_DIR}/ct-64.nrrd
      --olm ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
      --lcv -1200
      --lrv -1200
      --ucv 100
      --urv 100
)   
 
