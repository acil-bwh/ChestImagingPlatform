# SlicerExecutionModel
find_package(SlicerExecutionModel NO_MODULE REQUIRED GenerateCLP)
include(${SlicerExecutionModel_USE_FILE})

#FIND_PACKAGE( CIP REQUIRED )

macro(cipMacroBuildCLI)
  set(options
    NO_INSTALL VERBOSE
  )
  
  set (oneValueArgs
     NAME 
     LOGO_HEADER
  )
  
  set(multiValueArgs
    ADDITIONAL_TARGET_LIBRARIES
    ADDITIONAL_INCLUDE_DIRECTORIES
    SRCS
  )
  
  CMAKE_PARSE_ARGUMENTS(MY_CIP
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"  
    ${ARGN}
    )

  set ( INCLUDE_DIRECTORIES
    ${MY_CIP_ADDITIONAL_INCLUDE_DIRECTORIES}
    ${CIP_INCLUDE_DIRECTORIES}
   )
   
 
  set(TARGET_LIBRARIES
  ${MY_CIP_ADDITIONAL_TARGET_LIBRARIES}
  ${ITK_LIBRARIES}
  ${CIP_LIBRARIES}
  )

  
  if(${CIP_BUILD_CLI_EXECUTABLEONLY})
       set(PASS_EXECUTABLE_ONLY EXECUTABLE_ONLY)
  endif()
  
  SEMMacroBuildCLI(
       NAME ${MODULE_NAME}
       LOGO_HEADER ${MY_CIP_LOGO_HEADER}
       TARGET_LIBRARIES ${TARGET_LIBRARIES}
       INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORIES}
       ADDITIONAL_SRCS ${SRCS}
       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
       ${PASS_EXECUTABLE_ONLY}
  )  
 
 
  if(${BUILD_TESTING})  
		SET (INCLUDE_DIRECTORIES
	      ${INCLUDE_DIRECTORIES}
	      ${CIP_SOURCE_DIR}/CommandLineTools/Testing
	    )
	    # Default directories for input and ouput data for the tests
	   	SET (INPUT_DATA_DIR ${CIP_SOURCE_DIR}/Testing/Data/Input)
		  SET (OUTPUT_DATA_DIR ${CIP_BINARY_DIR}/CommandLineTools/Testing/Output)
	    file(MAKE_DIRECTORY "${OUTPUT_DATA_DIR}")

	  	INCLUDE_DIRECTORIES(${INCLUDE_DIRECTORIES})
	 	  ADD_EXECUTABLE(${MODULE_NAME}Test ./Testing/${MODULE_NAME}Test.cxx) 
	  	TARGET_LINK_LIBRARIES(${MODULE_NAME}Test ${MODULE_NAME}Lib ${TARGET_LIBRARIES})
	  	SET_TARGET_PROPERTIES(${MODULE_NAME}Test PROPERTIES LABELS ${MODULE_NAME} 
	          RUNTIME_OUTPUT_DIRECTORY ${CIP_BINARY_DIR}/CommandLineTools/Testing/bin
	    )
  endif()
 
 
endmacro()






  

