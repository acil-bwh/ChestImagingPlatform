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

if (USE_CYTHON)
  #if (APPLE)
     #Flag for Cython (See https://support.enthought.com/hc/en-us/articles/204469410-OS-X-GCC-Clang-and-Cython-in-10-9-Mavericks)
     #Due to the change to clang on OS X 1.9, you have to build against the old libs (libstdc++ and not the clang one - libc++).
  # 		set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -mmacosx-version-min=10.6 ${CMAKE_CXX_FLAGS}")
  #endif()


  	IF ( NOT CIP_PYTHON_DIR ) # probably built by SlicerCIP
    	SET ( CIP_PYTHON_CMD ${SLICER_PYTHON_CMD} ) # use Slicer Python
	ENDIF()
	MESSAGE("Wrapping ChestConventions with Python in ${CIP_PYTHON_CMD} and flags ${CMAKE_CXX_FLAGS}")

	#Copy .pyx file to build directory and execute there python
	FILE (COPY ${CMAKE_CURRENT_SOURCE_DIR} 
		DESTINATION ${CIP_PYTHON_INSTALL_FOLDER}
		FILES_MATCHING PATTERN "wrap_ChestConventions.pyx")
	
	CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/setup_ChestConventions.py.in 
	  ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py @ONLY )

	ADD_CUSTOM_TARGET ( ${PROJECT_NAME} ALL
	  COMMAND ${CIP_PYTHON_CMD} ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py build_ext -b ${CMAKE_CURRENT_BINARY_DIR}/
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
	ADD_DEPENDENCIES ( ${PROJECT_NAME} ChestConventions )
else()
	MESSAGE("ChestConventions will not be wrapped. Enable USE_CYTHON to wrap it")	
endif()


IF ( WIN32 )
  #NIPYPE is not supported on Windows
  SET(PYTHON_MODULES 
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

