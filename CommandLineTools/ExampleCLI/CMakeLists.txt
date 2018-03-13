cmake_minimum_required(VERSION 2.6.0)

PROJECT ( ExampleCLI )

SET ( MODULE_NAME ExampleCLI )
SET( MODULE_SRCS ExampleCLI.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
        --compare
        ${BASELINE_DATA_DIR}/${TEST_NAME}_crop_ct_smooth.nrrd
        ${OUTPUT_DATA_DIR}/${TEST_NAME}_crop_ct_smooth.nrrd
    ModuleEntryPoint
    #  --input "/Users/jonieva/workshop/004_INSP_STD_20130517_4797_4703_DECAMP1.nrrd"
    #  --output "/Users/jonieva/workshop/004_INSP_STD_20130517_4797_4703_DECAMP1_smooth.nrrd"
      --input ${INPUT_DATA_DIR}/crop_ct.nrrd    # INPUT_DATA_DIR = ChestImagingPlatform-build/CIP-build/Testing/Data/Input
      --output ${OUTPUT_DATA_DIR}/${TEST_NAME}_crop_ct_smooth.nrrd   # OUTPUT_DATA_DIR = ChestImagingPlatform-build/CIP-build/CommandLineTools/Testing/Output
      --variance 25.0

)
