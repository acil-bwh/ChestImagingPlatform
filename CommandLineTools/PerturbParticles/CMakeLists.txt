CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(PerturbParticles )

set(MODULE_NAME PerturbParticles)
set(MODULE_SRCS )

FIND_PACKAGE ( VTK REQUIRED)

set(MODULE_TARGET_LIBRARIES
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
      ${BASELINE_DATA_DIR}/${TEST_NAME}_airway_particles.vtk
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_particles.vtk
    --comparePointTolerance 4
    ModuleEntryPoint
      --ip ${INPUT_DATA_DIR}/airway_particles.vtk
      --op ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_particles.vtk
      -m 2
)
