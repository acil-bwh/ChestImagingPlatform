cmake_minimum_required(VERSION 2.6.0)

PROJECT ( ExecuteSystemCommand )

SET ( MODULE_NAME ExecuteSystemCommand )
SET( MODULE_SRCS ExecuteSystemCommand.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    ModuleEntryPoint
      -c "echo hi there!"
)

#SET (TEST_NAME ${MODULE_NAME}_Failed_Test)
#CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
#    ModuleEntryPoint
#      -c "echohi there!"
#)