cmake_minimum_required(VERSION 2.6.0)

PROJECT ( GenerateMedianFilteredImage )

SET ( MODULE_NAME GenerateMedianFilteredImage )

SET( MODULE_SRCS
  GenerateMedianFilteredImage.cxx
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )
