CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( PerformMorphological )

set(MODULE_NAME PerformMorphological)
set(MODULE_SRCS  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Dilate_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_baseline.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_out.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_out.nrrd
      -d
      -r LeftLung
      -t Vessel
      --radz 0
)   
