CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterConnectedComponents )

SET ( MODULE_NAME FilterConnectedComponents )
SET ( MODULE_SRCS FilterConnectedComponents.cxx )

ADD_LIBRARY( FilterConnectedComponentsHelper FilterConnectedComponentsHelper.cxx )

TARGET_LINK_LIBRARIES( FilterConnectedComponentsHelper
  ${ITK_LIBRARIES} 
  CIPCommon
  CIPIO
)

SET ( MODULE_TARGET_LIBRARIES
 FilterConnectedComponentsHelper
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
      ${INPUT_DATA_DIR}/simple_lm_filtered1.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/simple_lm.nrrd
      --vol 
      -s 2 
      --regionPairVecInclude 0,2 
      --typePairVecInclude 2,3
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
)
