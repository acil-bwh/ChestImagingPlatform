CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ExtractParticlesFromChestRegionChestType )

SET ( MODULE_NAME ExtractParticlesFromChestRegionChestType )
SET ( MODULE_SRCS ExtractParticlesFromChestRegionChestType.cxx )

SET ( MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test_airway_particles)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareVTK 
      ${BASELINE_DATA_DIR}/${TEST_NAME}.vtk
      ${OUTPUT_DATA_DIR}/${TEST_NAME}.vtk
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/airway_particles.vtk
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}.vtk
      -r 0
      -t 2
)