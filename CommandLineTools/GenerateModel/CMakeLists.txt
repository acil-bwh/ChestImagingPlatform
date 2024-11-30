CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateModel )

set(MODULE_NAME GenerateModel)

set(MODULE_TARGET_LIBRARIES
  ${VTK_LIBRARIES}
  ${ITK_LIBRARIES}
  )

set(MODULE_SRCS )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
# Different baseline images because there seems to be slight differences between platforms
if (UNIX)
    if (APPLE)
        SET (IMAGE_NAME ${TEST_NAME}_simple_lm_OSX.vtk)
    else()
        SET (IMAGE_NAME ${TEST_NAME}_simple_lm_LINUX.vtk)
    endif()
else()
    SET (IMAGE_NAME ${TEST_NAME}_simple_lm_WIN.vtk)
endif()

CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
   --compareVTKPolyData
      ${BASELINE_DATA_DIR}/${IMAGE_NAME}
      ${OUTPUT_DATA_DIR}/${IMAGE_NAME}
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      -l 2
      -o ${OUTPUT_DATA_DIR}/${IMAGE_NAME}
)
