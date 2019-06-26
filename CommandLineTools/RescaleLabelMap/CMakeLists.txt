cmake_minimum_required(VERSION 2.8.0)

project(RescaleLabelMap )

set(MODULE_NAME RescaleLabelMap)
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
    --compare 	  
    ${BASELINE_DATA_DIR}/${TEST_NAME}_lm-32.nrrd
    ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/lm-64.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy.nrrd
     --down 2
      )
