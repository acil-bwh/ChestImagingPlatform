CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CropLung )

SET ( MODULE_NAME CropLung )
SET ( MODULE_SRCS CropLung.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    SRCS ${MODULE_SRCS}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    )


SET (RIGHTLUNG_TEST_NAME ${MODULE_NAME}_RightLung_Test)
CIP_ADD_TEST(NAME ${RIGHTLUNG_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compareCT
      ${BASELINE_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
      ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
    --compareLabelMap 
      ${BASELINE_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd
      ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd    
    ModuleEntryPoint
      --ict ${INPUT_DATA_DIR}/simple_ct.nrrd
      --ilm ${INPUT_DATA_DIR}/simple_lm.nrrd
      --cipr RightLung 
      --padding 2,2,2 
      -m 0 
      -v 1000 
      --oct ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
      --olm ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd       
    )


SET (LEFTLUNG_TEST_NAME ${MODULE_NAME}_LeftLung_Test)   
CIP_ADD_TEST_MIDAS( NAME ${LEFTLUNG_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test> MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} 
      --compareCT 
         MIDAS{${LEFTLUNG_TEST_NAME}_ct.nrrd.md5}
         ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_ct.nrrd
      --compareLabelMap 
         MIDAS{${LEFTLUNG_TEST_NAME}_lm.nrrd.md5}
         ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_lm.nrrd
       ModuleEntryPoint
        --ict ${INPUT_DATA_DIR}/simple_ct.nrrd
        --ilm ${INPUT_DATA_DIR}/simple_lm.nrrd
        --cipr LeftLung 
        --cipp 2,2,2 
        -m 0 
        -v 1000 
        --oct ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_ct.nrrd
        --olm ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_lm.nrrd      
)


    
# SET (CHECKPOINT_TEST_NAME ${MODULE_NAME}_CheckPoint_Test)
 
# CIP_ADD_TEST(NAME ${CHECKPOINT_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>   
#     CheckPointEntryPoint    	 	
# )

# SET (TEST_NAME ${MODULE_NAME}_Test)
#  CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#     --compare 
#       ${BASELINE_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#       ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#     --compare 
#       ${BASELINE_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
#       ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
#     ModuleEntryPoint
#       -i ${INPUT_DATA_DIR}/dummy_ct.nrrd
#       -l ${INPUT_DATA_DIR}/dummy_lm.nrrd
#       -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
#       --opl ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
# )
