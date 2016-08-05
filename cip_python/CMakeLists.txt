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

if (NOT CIP_PYTHON_INSTALL_FOLDER)
	SET (CIP_PYTHON_INSTALL_FOLDER  ${CMAKE_BINARY_DIR}/cip_python)
endif()

IF ( WIN32 )
  #NIPYPE is not supported on Windows
  SET(PYTHON_MODULES
    common
	classification
	input_output
	particles
	phenotypes
	qualitycontrol
	registration
	segmentation
	utils
	visualization
  )
ELSE ( )
  SET(PYTHON_MODULES
    common
	classification
	input_output
	nipype
	particles
	phenotypes
	qualitycontrol
	registration
	segmentation
	utils
	visualization
  )
ENDIF ( )

FILE (INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
		DESTINATION ${CIP_PYTHON_INSTALL_FOLDER}
		)

FOREACH (PYTHON_MODULE ${PYTHON_MODULES})
	# Include current tests in order that can be identified by ctest
	IF ( CIP_BUILD_TESTING )
		SUBDIRS(${PYTHON_MODULE})
	ENDIF ( )

	#Perform Module installation
	SET (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_MODULE})
	SET (TARGET_DIR ${CIP_PYTHON_INSTALL_FOLDER})
	FILE (INSTALL ${SOURCE_DIR} 
	  DESTINATION ${TARGET_DIR}
	  FILES_MATCHING PATTERN "*.py")
ENDFOREACH ( )
