CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FitLobeSurfaceModelsToParticleData )

SET ( MODULE_NAME FitLobeSurfaceModelsToParticleData )
SET ( MODULE_SRCS FitLobeSurfaceModelsToParticleData.cxx )

SET ( MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
  )



cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

# TODO: WRITE TEST
#SET (TEST_NAME ${MODULE_NAME}_Test)
#CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#    --compare 
#      ${BASELINE_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#    --compare 
#      ${BASELINE_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
#      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
#    ModuleEntryPoint
#      -i ${INPUT_DATA_DIR}/dummy_ct.nrrd
#      -l ${INPUT_DATA_DIR}/dummy_lm.nrrd
#      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#      --opl ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
#)
