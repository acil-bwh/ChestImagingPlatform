
set(proj ITKv5)

# Set dependency list
set(${proj}_DEPENDENCIES "zlib" ${VTK_EXTERNAL_NAME})

set(CIP_BUILD_DICOM_SUPPORT OFF)

if(CIP_BUILD_DICOM_SUPPORT)
  list(APPEND ${proj}_DEPENDENCIES DCMTK)
endif()

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
  unset(ITK_DIR CACHE)
  #find_package(ITK 4 COMPONENTS ${${CMAKE_PROJECT_NAME}_ITK_COMPONENTS} REQUIRED NO_MODULE)
  find_package(ITK 4.6 REQUIRED NO_MODULE)
endif()

# Sanity checks
if(DEFINED ITK_DIR AND NOT EXISTS ${ITK_DIR})
  message(FATAL_ERROR "ITK_DIR variable is defined but corresponds to non-existing directory")
endif()

if(NOT DEFINED ITK_DIR AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})

  if(NOT DEFINED git_protocol)
      set(git_protocol "git")
  endif()

  set(ITKv5_REPOSITORY ${git_protocol}://github.com/Slicer/ITK.git)


  set(ITKv5_GIT_TAG 35e6f546438557f22e66db25e066499637890214)  # v5.0.1, slicer-v5.0.1-2019-07-23-aaa9989c5f

 
  set(CIP_ITKV3_COMPATIBILITY OFF) # to match the default setting of Slicer

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  if(NOT CIP_ITKV3_COMPATIBILITY AND CMAKE_CL_64) # follow the same logic as in Slicer
    # enables using long long type for indexes and size on platforms
    # where long is only 32-bits (msvc)
    set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DITK_USE_64BITS_IDS:BOOL=ON
      )
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY ${ITKv5_REPOSITORY}
    GIT_TAG ${ITKv5_GIT_TAG}
    SOURCE_DIR ${proj}
    BINARY_DIR ${proj}-build
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      -DITK_INSTALL_ARCHIVE_DIR:PATH=${CIP_INSTALL_LIB_DIR}
      -DITK_INSTALL_LIBRARY_DIR:PATH=${CIP_INSTALL_LIB_DIR}
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DITK_LEGACY_REMOVE:BOOL=ON
      #-DITKV3_COMPATIBILITY:BOOL=${CIP_ITKV3_COMPATIBILITY}
      -DITKV3_COMPATIBILITY:BOOL=OFF
      -DITK_BUILD_DEFAULT_MODULES:BOOL=ON
      -DModule_ITKReview:BOOL=ON
      -DModule_MGHIO:BOOL=ON
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DITK_INSTALL_NO_DEVELOPMENT:BOOL=ON
      -DKWSYS_USE_MD5:BOOL=ON # Required by SlicerExecutionModel
      -DITK_WRAPPING:BOOL=OFF #${BUILD_SHARED_LIBS} ## HACK:  QUICK CHANGE
      -DVTK_DIR:PATH=${VTK_DIR}
      -DModule_ITKVtkGlue:BOOL=ON
      -DITK_USE_CONCEPT_CHECKING:BOOL=ON
      # DCMTK
      -DITK_USE_SYSTEM_DCMTK:BOOL=ON
      -DDCMTK_DIR:PATH=${DCMTK_DIR}
      -DModule_ITKIODCMTK:BOOL=${CIP_BUILD_DICOM_SUPPORT}
      # ZLIB
      -DITK_USE_SYSTEM_ZLIB:BOOL=ON
      -DZLIB_ROOT:PATH=${ZLIB_ROOT}
      -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
      -DZLIB_LIBRARY:FILEPATH=${ZLIB_LIBRARY}
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    INSTALL_COMMAND ""
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(ITK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(
  VARS ITK_DIR:PATH
  LABELS "FIND_PACKAGE"
  )

