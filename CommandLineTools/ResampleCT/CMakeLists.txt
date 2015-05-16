cmake_minimum_required(VERSION 2.8.0)

project(ResampleCT )

set(MODULE_NAME ResampleCT)
set(MODULE_SRCS )

find_package(SlicerExecutionModel REQUIRED)
include(${SlicerExecutionModel_USE_FILE})

 FIND_PACKAGE( ITK REQUIRED )

 FIND_PACKAGE( CIP REQUIRED )

set(MODULE_INCLUDE_DIRECTORIES
  ${CIP_INCLUDE_DIRECTORIES}
  )

set(MODULE_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  ${CIP_LIBRARIES}
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCT 
      ${INPUT_DATA_DIR}/ct-64-transformed.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
    --compareIntensityTolerance 4	
    ModuleEntryPoint
      --ict ${INPUT_DATA_DIR}/ct-64.nrrd
      --destct ${INPUT_DATA_DIR}/ct-64-transformed.nrrd
      --itx ${INPUT_DATA_DIR}/TranslationRotation.tfm
      --dim 3		
      --oct ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
)