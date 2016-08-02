CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ConvertDicom )

set(MODULE_NAME ConvertDicom)

set(MODULE_SRCS )
  
cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCT 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_simple_dicom.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_dicom.nrrd
    ModuleEntryPoint
      --dir ${INPUT_DATA_DIR}/simple_dicom
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_dicom.nrrd
)

