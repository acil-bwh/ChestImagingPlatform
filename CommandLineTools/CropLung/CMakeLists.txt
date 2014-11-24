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


# SET (LEFTLUNG_TEST_NAME ${MODULE_NAME}_LeftLung_Test)
# add_test(NAME ${LEFTLUNG_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#     --compare 
#     	${INPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_ct.nrrd
#     	${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_ct.nrrd
#     --compare 
#     	${INPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_lm.nrrd
#     	${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_lm.nrrd    
#     ModuleEntryPoint
#     	-i ${INPUT_DATA_DIR}/simple_ct.nrrd
#     	-l ${INPUT_DATA_DIR}/simple_lm.nrrd
#     	-r LeftLung 
#     	-p 2,2,2 
#     	-m 0 
#     	-v 1000 
#     	-o ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_ct.nrrd
#     	--opl ${OUTPUT_DATA_DIR}/${LEFTLUNG_TEST_NAME}_lm.nrrd     	
#     )
    
# SET (RIGHTLUNG_TEST_NAME ${MODULE_NAME}_RightLung_Test)
# add_test(NAME ${RIGHTLUNG_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#     --compare 
#     	${INPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
#     	${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
#     --compare 
#     	${INPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd
#     	${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd    
#     ModuleEntryPoint
#     	-i ${INPUT_DATA_DIR}/simple_ct.nrrd
#     	-l ${INPUT_DATA_DIR}/simple_lm.nrrd
#     	-r RIGHTLung 
#     	-p 2,2,2 
#     	-m 0 
#     	-v 1000 
#     	-o ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_ct.nrrd
#     	--opl ${OUTPUT_DATA_DIR}/${RIGHTLUNG_TEST_NAME}_lm.nrrd     	
#     )
    
# SET (CHECKPOINT_TEST_NAME ${MODULE_NAME}_CheckPoint_Test)
 
# add_test(NAME ${CHECKPOINT_TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>   
#     CheckPointEntryPoint    	 	
# )

SET (TEST_NAME ${MODULE_NAME}_Test)
 add_test(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare 
      ${INPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
    --compare 
      ${INPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
    ModuleEntryPoint
      -i ${INPUT_DATA_DIR}/dummy_ct.nrrd
      -l ${INPUT_DATA_DIR}/dummy_lm.nrrd
      -o ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_ct.nrrd
      --opl ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
)