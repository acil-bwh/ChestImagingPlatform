# SlicerExecutionModel
find_package(SlicerExecutionModel NO_MODULE REQUIRED GenerateCLP)
include(${SlicerExecutionModel_USE_FILE})

FIND_PACKAGE( CIP REQUIRED )

#cipMacroBuildCLI.cmake
macro(cipMacroBuildCLI)
  set(options
    NO_INSTALL VERBOSE
  )
  
  set (oneValueArgs
     NAME 
     LOGO_HEADER
  )
  
  set(multiValueArgs
    TARGET_LIBRARIES
    SRCS
  )
  
  CMAKE_PARSE_ARGUMENTS(MY_CIP
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"  
    ${ARGN}
    )

  set ( INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common
    ${CMAKE_SOURCE_DIR}/Utilities
    ${LIBXML2_INCLUDE_DIR}
   )

  
  if(${CIP_BUILD_CLI_EXECUTABLEONLY})
       set(PASS_EXECUTABLE_ONLY EXECUTABLE_ONLY)
  endif()
  
  SEMMacroBuildCLI(
       NAME ${MODULE_NAME}
       LOGO_HEADER ${MY_CIP_LOGO_HEADER}
       TARGET_LIBRARIES ${MY_CIP_TARGET_LIBRARIES}
       INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORIES}
       ADDITIONAL_SRCS ${SRCS}
       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       ${PASS_EXECUTABLE_ONLY}
  )
endmacro()