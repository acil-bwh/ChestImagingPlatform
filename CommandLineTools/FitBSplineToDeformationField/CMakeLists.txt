cmake_minimum_required(VERSION 2.6.0)

PROJECT ( FitBSplineToDeformationField )

SET ( MODULE_NAME FitBSplineToDeformationField )
SET( MODULE_SRCS FitBSplineToDeformationField.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test1)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCSV 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_bspline20.tfm
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_bspline20.tfm
    ModuleEntryPoint
    -i ${INPUT_DATA_DIR}/ct-displacementField.nrrd
    -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_bspline20.tfm
     -n 20
)

SET (TEST_NAME ${MODULE_NAME}_Test2)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCSV 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_bspline40.tfm
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_bspline40.tfm
    ModuleEntryPoint
    -i ${INPUT_DATA_DIR}/ct-displacementField.nrrd
    -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_bspline40.tfm
    -n 40
)