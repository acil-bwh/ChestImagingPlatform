CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ComputeFeatureStrength )

SET ( MODULE_NAME ComputeFeatureStrength )
SET ( MODULE_SRCS ComputeFeatureStrength.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )
    
SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_vessel_frangi.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_vessel_frangi.nrrd
    ModuleEntryPoint
    -m Frangi 
    -f RidgeLine 
    --C 200 
    -i ${INPUT_DATA_DIR}/vessel.nrrd
    --std 0.6,5,4
    -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_vessel_frangi.nrrd
)

SET (TEST_NAME ${MODULE_NAME}_Test2)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${BASELINE_DATA_DIR}/${TEST_NAME}_airway_frangi.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_frangi.nrrd
    ModuleEntryPoint
    -m Frangi 
    -f ValleyLine 
    --C 100
    -i ${INPUT_DATA_DIR}/airway.nrrd
    --std 0.6,5,4
    -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_airway_frangi.nrrd
)


