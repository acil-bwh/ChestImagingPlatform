CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ReadVTKWriteNRRDs )

SET ( MODULE_NAME ReadVTKWriteNRRDs )
SET ( MODULE_SRCS ReadVTKWriteNRRDs.cxx )


SET ( MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
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
      ${BASELINE_DATA_DIR}/${TEST_NAME}_h0.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_h0.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_h1.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_h1.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_h2.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_h2.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_hess.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_hess.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_hevec0.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_hevec0.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_hevec1.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_hevec1.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_hevec2.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_hevec2.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_hmode.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_hmode.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_pass.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_pass.nrrd
    --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_val.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_val.nrrd
   --compare
      ${BASELINE_DATA_DIR}/${TEST_NAME}_ChestRegionChestType.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_ChestRegionChestType.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/vessel_particles.vtk
      --prefix ${OUTPUT_DATA_DIR}/${TEST_NAME}_
)
