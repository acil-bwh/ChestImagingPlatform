PROJECT ( CIP_PYTHON )

# to build from SlicerCIP superbuild
IF ( NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY )
  SET ( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH} )
ENDIF()

IF ( WIN32 )
  SET ( CIP_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE} )
  SET ( CIP_PYTHON_CMD ${CIP_PYTHON_DIR}/python.exe )
ELSE ( WIN32 )
  SET ( CIP_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
  SET ( CIP_PYTHON_CMD ${CIP_PYTHON_DIR}/bin/python )
ENDIF ( WIN32 )

if (CIP_WRAPCHESTCONVENTIONS)
  	IF ( NOT CIP_PYTHON_DIR ) # probably built by SlicerCIP
    	SET ( CIP_PYTHON_CMD ${SLICER_PYTHON_CMD} ) # use Slicer Python
	ENDIF()
	MESSAGE("Wrapping ChestConventions with Python in ${CIP_PYTHON_CMD}")
	
	CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/setup_ChestConventions.py.in 
	  ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py @ONLY )

	ADD_CUSTOM_TARGET ( ${PROJECT_NAME} ALL
	  COMMAND ${CIP_PYTHON_CMD} ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py build_ext -b ${CMAKE_CURRENT_BINARY_DIR}/
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
	ADD_DEPENDENCIES ( ${PROJECT_NAME} ChestConventions )
else()
	MESSAGE("ChestConventions will not be wrapped. Enable CIP_WRAPCHESTCONVENTIONS to wrap it")	
endif()

FILE (COPY ${CMAKE_CURRENT_SOURCE_DIR} 
	DESTINATION ${CMAKE_BINARY_DIR}
	FILES_MATCHING PATTERN "*.py")

# Include current tests in order that can be identified by ctest
IF ( CIP_BUILD_TESTING )
 SUBDIRS ( particles )
 SUBDIRS ( phenotypes )
 SUBDIRS ( utils )
 SUBDIRS ( nipype )
 SUBDIRS ( segmentation )
 SUBDIRS ( classification )
 SUBDIRS ( input_output )
ENDIF( CIP_BUILD_TESTING ) 
