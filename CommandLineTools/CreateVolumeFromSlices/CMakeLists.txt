CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CreateVolumeFromSlices )

set(MODULE_NAME CreateVolumeFromSlices)

set(MODULE_SRCS CreateVolumeFromSlices.cxx)
  
cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

#SET (TEST_NAME ${MODULE_NAME}_Test)
#CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#    --compareCT 
#      ${BASELINE_DATA_DIR}/${TEST_NAME}_simple_volume.nrrd
#      ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_volume.nrrd
#    ModuleEntryPoint
#      -p ${INPUT_DATA_DIR}/slices/%s%02d.bmp
#      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_simple_volume.nrrd
#)

