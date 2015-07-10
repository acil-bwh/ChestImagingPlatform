CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RegionTypeLocationsToROIVolume )

set(MODULE_NAME RegionTypeLocationsToROIVolume )
set(MODULE_SRCS )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )


SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_baseline.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_test.nrrd
    ModuleEntryPoint
      --xr 1
      --yr 1
      --zr 0
      --ict ${INPUT_DATA_DIR}/simple_ct.nrrd
      --ip  ${INPUT_DATA_DIR}/simple_regionAndTypePoints.csv
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_test.nrrd
      --seg
)
