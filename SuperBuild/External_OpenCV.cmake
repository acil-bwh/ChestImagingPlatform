
set(proj OpenCV)

# Set dependency list
set(${proj}_DEPENDENCIES "")

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  unset(OpenCV_DIR CACHE)
  find_package(OpenCV REQUIRED)
endif()

# Sanity checks
if(DEFINED OpenCV_DIR AND NOT EXISTS ${OpenCV_DIR})
  message(FATAL_ERROR "OpenCV_DIR variable is defined but corresponds to non-existing directory")
endif()

if(NOT DEFINED OpenCV_DIR AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})

  if(NOT DEFINED git_protocol)
    set(git_protocol "git")
  endif()

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "${git_protocol}://github.com/Itseez/opencv.git"
    GIT_TAG "c12243cf4fccf5df7b0270a32883986b373dca7b" # 3.0.0 tag
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      ## CXX should not be needed, but it a cmake default test
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DVTK_DIR:PATH=${VTK_DIR}
      -DVTK_VERSION_MAJOR:STRING=${VTK_VERSION_MAJOR}
#      -DPYTHON_EXECUTABLE:FILEPATH=${CIP_PYTHON_EXECUTABLE}
#      -DPYTHON_INCLUDE_DIR:PATH=${CIP_PYTHON_INCLUDE_DIR}
#      -DPYTHON_LIBRARY:FILEPATH=${CIP_PYTHON_LIBRARY}
      -DPYTHON2_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
      -DPYTHON2_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
      -DPYTHON2_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
#      -DPYTHON2_PACKAGES_PATH:PATH=${CIP_PYTHON_PACKAGES_PATH}
      ##Remove OpenCV packages that are not needed
      -DBUILD_opencv_java:BOOLEAN=OFF
      -DBUILD_opencv_video:BOOLEAN=OFF
      -DBUILD_opencv_videoio:BOOLEAN=OFF
      -DBUILD_opencv_videostab:BOOLEAN=OFF
      -DBUILD_opencv_calib3d:BOOLEAN=OFF
      -DBUILD_opencv_viz:BOOLEAN=OFF
      ##Disable VTK in opencv. There is not need now to do VTK vis in opencv
      -DWITH_VTK:BOOLEAN=OFF
      #Disable Python for now
      -DBUILD_opencv_python2:BOOLEAN=OFF
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(OpenCV_DIR ${EP_BINARY_DIR})
else()
  # The project is provided using zlib_DIR, nevertheless since other project may depend on zlib,
  # let's add an 'empty' one
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(
  VARS OpenCV_DIR:PATH
  LABELS "FIND_PACKAGE"
  )

