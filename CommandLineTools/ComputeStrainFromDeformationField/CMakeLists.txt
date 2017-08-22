cmake_minimum_required(VERSION 2.6.0)

PROJECT ( ComputeStrainFromDeformationField )

SET ( MODULE_NAME ComputeStrainFromDeformationField )
SET( MODULE_SRCS ComputeStrainFromDeformationField.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

#SET (TEST_NAME ${MODULE_NAME}_Test)
#CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#    --compareCT 
#      ${BASELINE_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
#      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ct-64.nrrd
#    ModuleEntryPoint
#      -i ${INPUT_DATA_DIR}/ct-64.nrrd
#      —-outputL1 ${OUTPUT_DATA_DIR}/${TEST_NAME}_strainL1.nrrd
#      —-outputL2 ${OUTPUT_DATA_DIR}/${TEST_NAME}_strainL2.nrrd
#      —-outputL3 ${OUTPUT_DATA_DIR}/${TEST_NAME}_strainL3.nrrd
#      -t Almansi
#)