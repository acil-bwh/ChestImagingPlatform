PROJECT ( CIP_PYTHON )

# to build from SlicerCIP superbuild
IF ( NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY )
  SET ( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH} )
ENDIF()

IF ( WIN32 )
  SET ( CIP_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE} )
ELSE ( WIN32 )
  SET ( CIP_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
ENDIF ( WIN32 )

if (NOT CIP_PYTHON_MODULES_INSTALL_FOLDER)
	SET (CIP_PYTHON_MODULES_INSTALL_FOLDER  ${CMAKE_BINARY_DIR}/cip_python)
endif()

IF ( WIN32 )
  # nipype / tensorflow are not supported on Windows
  SET(PYTHON_MODULES
    common
    classification
    #dcnn
    input_output
    particles
    phenotypes
    quality_control
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
    dcnn
    nipype
    particles
    phenotypes
    quality_control
    registration
    segmentation
    utils
    visualization
  )
ENDIF ( )
FILE (INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
		DESTINATION ${CIP_PYTHON_MODULES_INSTALL_FOLDER}
		)
if (USE_CYTHON)
  #if (APPLE)
     #Flag for Cython (See https://support.enthought.com/hc/en-us/articles/204469410-OS-X-GCC-Clang-and-Cython-in-10-9-Mavericks)
     #Due to the change to clang on OS X 1.9, you have to build against the old libs (libstdc++ and not the clang one - libc++).
  #             set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ -mmacosx-version-min=10.6 ${CMAKE_CXX_FLAGS}")
  #endif()
        MESSAGE("Wrapping GraphCutsOptimization with Python in ${PYTHON_EXECUTABLE} and flags ${CMAKE_CXX_FLAGS}")

        CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/utils/gco_python/setup.py.in
          ${CMAKE_CURRENT_SOURCE_DIR}/utils/gco_python/setup.py @ONLY )

        ADD_CUSTOM_TARGET ( ${PROJECT_NAME} ALL
          COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/gco_python/setup.py build_ext -b ${CMAKE_CURRENT_BINARY_DIR}/utils
          #WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
else()
        MESSAGE("GraphCutsOptimization will not be wrapped. Enable USE_CYTHON to wrap it")
endif()


FOREACH (PYTHON_MODULE ${PYTHON_MODULES})
	# Include current tests in order that can be identified by ctest
	IF ( CIP_BUILD_TESTING )
		SUBDIRS(${PYTHON_MODULE})
	ENDIF ( )

	#Perform Module installation
	SET (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_MODULE})
	SET (TARGET_DIR ${CIP_PYTHON_MODULES_INSTALL_FOLDER})
	FILE (INSTALL ${SOURCE_DIR}
	  DESTINATION ${TARGET_DIR}
	  FILES_MATCHING PATTERN "*.py")


	if (EXISTS ${SOURCE_DIR}/data/baseline)
		# Include all the files in the baseline
		FILE(INSTALL ${SOURCE_DIR}/data/baseline
				DESTINATION ${TARGET_DIR}/${PYTHON_MODULE}/data
				FILES_MATCHING PATTERN "*.*"
		)
	endif()
ENDFOREACH ( )
