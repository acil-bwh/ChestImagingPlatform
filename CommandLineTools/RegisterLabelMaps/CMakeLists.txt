CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(RegisterLabelMaps)

set(MODULE_NAME RegisterLabelMaps)
set(MODULE_SRCS itkCIPMergeChestLabelMapsImageFilter2D3D.cxx )


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
    --compareLabelMapDice 
      ${INPUT_DATA_DIR}/lm-64.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
    --diceTolerance 99	
    ModuleEntryPoint
      --mlm ${INPUT_DATA_DIR}/lm-64-transformed.nrrd
      --flm ${INPUT_DATA_DIR}/lm-64.nrrd
      --olm ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
      --otx ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_tx.tfm	
      --dim 3
)
