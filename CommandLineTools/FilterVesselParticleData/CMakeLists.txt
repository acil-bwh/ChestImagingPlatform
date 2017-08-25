CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterVesselParticleData )

SET ( MODULE_NAME FilterVesselParticleData )
SET ( MODULE_SRCS FilterVesselParticleData.cxx )

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
      -i ${INPUT_DATA_DIR}/case_vessel_particles.vtk
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_case_vessel_particles.vtk
      -d 2
      --size 60
)
