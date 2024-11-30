cmake_minimum_required(VERSION 2.6.0)

PROJECT ( ComputeAirwayWallFromImages )

SET ( MODULE_NAME ComputeAirwayWallFromImages )
SET( MODULE_SRCS ComputeAirwayWallFromImages.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )


SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCSV 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ZCAirwayPhenotypes.csv
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_FWHMAirwayPhenotypes.csv
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/airway_slices.nrrd
      -m ZC
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_ZCAirwayPhenotypes.csv
)

SET (TEST_NAME ${MODULE_NAME}_Test2)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCSV 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_FWHMAirwayPhenotypes.csv
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_FWHMAirwayPhenotypes.csv
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/airway_slices.nrrd
      -m FWHM
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_FWHMAirwayPhenotypes.csv
)