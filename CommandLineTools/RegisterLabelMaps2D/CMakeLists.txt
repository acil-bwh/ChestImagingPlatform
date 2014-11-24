CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(RegisterLabelMaps2D)
set(MODULE_SRCS )

set(MODULE_NAME RegisterLabelMaps2D)

find_package(LibXml2 REQUIRED)

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
add_test(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${INPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
    --compare 
      ${INPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/dummy_ct.nrrd
      -l ${INPUT_DATA_DIR}/dummy_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
      --opl ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
)    