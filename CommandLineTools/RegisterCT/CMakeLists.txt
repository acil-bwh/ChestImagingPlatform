CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(RegisterCT)

set(MODULE_NAME RegisterCT)
set(MODULE_SRCS )

find_package(LibXml2)

 set (UUID_LIBS -luuid)
 
set(MODULE_INCLUDE_DIRECTORIES
  ${LIBXML2_INCLUDE_DIR}
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
    --compareLabelMap 
      ${INPUT_DATA_DIR}/ct-64.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
    --compareIntensityTolerance 4	
    ModuleEntryPoint
      -f ${INPUT_DATA_DIR}/ct-64.nrrd
      -m ${INPUT_DATA_DIR}/ct-64-transformed.nrrd
      --outputTransform ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_tx.tfm
      --dimension 3		
      --resampled ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
)
