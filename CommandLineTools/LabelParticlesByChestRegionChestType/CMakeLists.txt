CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LabelParticlesByChestRegionChestType )

SET ( MODULE_NAME LabelParticlesByChestRegionChestType )
SET ( MODULE_SRCS LabelParticlesByChestRegionChestType.cxx )

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
    --compareVTKPolyData
      ${BASELINE_DATA_DIR}/${TEST_NAME}_case_vessel_particles.vtk
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_case_vessel_particles.vtk
    ModuleEntryPoint
      --ip ${INPUT_DATA_DIR}/case_vessel_particles.vtk
      --ilm ${INPUT_DATA_DIR}/lm-64.nrrd
      --op ${OUTPUT_DATA_DIR}/${TEST_NAME}_case_vessel_particles.vtk
)   
