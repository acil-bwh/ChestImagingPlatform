CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(RegisterCT)

set(MODULE_NAME RegisterCT)
set(MODULE_SRCS )


set (UUID_LIBS -luuid)
 
set(MODULE_INCLUDE_DIRECTORIES
  ${VTK_LIBXML_INCLUDE_DIR}
  )

set(MODULE_TARGET_LIBRARIES
  ${LIBXML2_LIBRARIES}
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
      ${INPUT_DATA_DIR}/ct-64.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
    --compareIntensityTolerance 15
    ModuleEntryPoint
      --fct ${INPUT_DATA_DIR}/ct-64.nrrd
      --mct ${INPUT_DATA_DIR}/ct-64-transformed.nrrd
      --otx ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_tx.tfm
      --dim 3		
      --oct ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
)
