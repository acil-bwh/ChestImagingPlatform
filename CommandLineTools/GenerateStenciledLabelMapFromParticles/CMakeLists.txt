CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateStenciledLabelMapFromParticles )

SET ( MODULE_NAME GenerateStenciledLabelMapFromParticles )
SET ( MODULE_SRCS GenerateStenciledLabelMapFromParticles.cxx )

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
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_airway_particles.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_particles.nrrd
    ModuleEntryPoint
      --ip ${INPUT_DATA_DIR}/airway_particles.vtk
      --ilm ${INPUT_DATA_DIR}/airway.nrrd
      --olm ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_particles.nrrd
      -c
      -a
      --height 0.9
      --radius 2
)
