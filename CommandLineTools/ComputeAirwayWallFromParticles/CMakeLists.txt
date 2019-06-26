cmake_minimum_required(VERSION 2.6.0)

PROJECT ( ComputeAirwayWallFromParticles )

SET ( MODULE_NAME ComputeAirwayWallFromParticles )
SET( MODULE_SRCS ComputeAirwayWallFromParticles.cxx )

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
    ModuleEntryPoint
      --ip ${INPUT_DATA_DIR}/airway_particles.vtk
      --ict ${INPUT_DATA_DIR}/airwaygauss.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_particles.vtk
      -n 10
      -m ZC
      --wth 250
)
