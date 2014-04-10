CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(GenerateInteractiveLobeSegmentationFissureMask )

SET ( MODULE_NAME GenerateInteractiveLobeSegmentationFissureMask )

SET ( MODULE_SRCS GenerateInteractiveLobeSegmentationFissureMask.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )