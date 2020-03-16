#-----------------------------------------------------------------------------
enable_language(C)
enable_language(CXX)

#-----------------------------------------------------------------------------
enable_testing()
include(CTest)

#-----------------------------------------------------------------------------
#include(${CMAKE_CURRENT_SOURCE_DIR}/Common.cmake)

# start Common.cmake
include(CMakeDependentOption)

option(${PRIMARY_PROJECT_NAME}_INSTALL_DEVELOPMENT "Install development support include and libraries for external packages." OFF)
mark_as_advanced(${PRIMARY_PROJECT_NAME}_INSTALL_DEVELOPMENT)

option(${PRIMARY_PROJECT_NAME}_USE_QT "Find and use Qt with VTK to build GUI Tools" OFF)
mark_as_advanced(${PRIMARY_PROJECT_NAME}_USE_QT)

set(ITK_VERSION_MAJOR 4 CACHE STRING "Choose the expected ITK major version to build, only version 4 allowed. (testing 5)")
set_property(CACHE ITK_VERSION_MAJOR PROPERTY STRINGS "4" "5")

set(VTK_VERSION_MAJOR 8 CACHE STRING "Choose the expected VTK major version to build. At least version 7 is strongly recommended.")
set_property(CACHE VTK_VERSION_MAJOR PROPERTY STRINGS "9" "8" "7" "6")


#-----------------------------------------------------------------------------
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  if (CMAKE_CONFIGURATION_TYPES)
    list(GET CMAKE_CONFIGURATION_TYPES 0 BT_TMP) 
  message("Setting build type to the first active configuration type because it was not specified: ${BT_TMP}" )
  else()
    set(BT_TMP "Release")
  message(STATUS "Setting build type to 'Release' as none was specified and there are not CMAKE_CONFIGURATION_TYPES")
  endif()  
  set(CMAKE_BUILD_TYPE ${BT_TMP} CACHE STRING "Choose the type of build." FORCE)  
endif()
#if(NOT CMAKE_CONFIGURATION_TYPES)
  # Set the possible values of build type for cmake-gui
#  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
#else()
#  MESSAGE("CONFIG TYPES: ${CMAKE_CONFIGURATION_TYPES}")
#  list(REMOVE_ITEM CMAKE_CONFIGURATION_TYPES "Release") 
#  list(INSERT CMAKE_CONFIGURATION_TYPES 0 "Release")
#  MESSAGE("CONFIG TYPES after: ${CMAKE_CONFIGURATION_TYPES}")
#endif()

#-----------------------------------------------------------------------------
# Update CMake module path
#------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
  ${${PROJECT_NAME}_SOURCE_DIR}/CMake
  ${${PROJECT_NAME}_BINARY_DIR}/CMake
  ${CMAKE_MODULE_PATH}
  )

#-----------------------------------------------------------------------------
# Sanity checks
#------------------------------------------------------------------------------
include(PreventInSourceBuilds)
include(PreventInBuildInstalls)

#-----------------------------------------------------------------------------
# Platform check
#-----------------------------------------------------------------------------
set(PLATFORM_CHECK true)
if(PLATFORM_CHECK)
  # See CMake/Modules/Platform/Darwin.cmake)
  #   6.x == Mac OSX 10.2 (Jaguar)
  #   7.x == Mac OSX 10.3 (Panther)
  #   8.x == Mac OSX 10.4 (Tiger)
  #   9.x == Mac OSX 10.5 (Leopard)
  #  10.x == Mac OSX 10.6 (Snow Leopard)
  if (DARWIN_MAJOR_VERSION LESS "9")
    message(FATAL_ERROR "Only Mac OSX >= 10.5 are supported !")
  endif()
endif()

#-----------------------------------------------------------------------------
if(NOT COMMAND SETIFEMPTY)
  macro(SETIFEMPTY)
    set(KEY ${ARGV0})
    set(VALUE ${ARGV1})
    if(NOT ${KEY})
      set(${ARGV})
    endif()
  endmacro()
endif()

#-----------------------------------------------------------------------------
SETIFEMPTY(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
SETIFEMPTY(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
SETIFEMPTY(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

#-----------------------------------------------------------------------------
SETIFEMPTY(CMAKE_INSTALL_LIBRARY_DESTINATION lib)
SETIFEMPTY(CMAKE_INSTALL_ARCHIVE_DESTINATION lib)
SETIFEMPTY(CMAKE_INSTALL_RUNTIME_DESTINATION bin)

#-------------------------------------------------------------------------
# Augment compiler flags
#-------------------------------------------------------------------------
#include(ITKSetStandardCompilerFlags)

#------------------------------------------------------------------------
# Check for clang -- c++11 necessary for boost
#------------------------------------------------------------------------
if("${CMAKE_CXX_COMPILER}${CMAKE_CXX_COMPILER_ARG1}" MATCHES ".*clang.*")
  set(CMAKE_COMPILER_IS_CLANGXX ON CACHE BOOL "compiler is CLang")
endif()

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${ITK_REQUIRED_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ITK_REQUIRED_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ITK_REQUIRED_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ITK_REQUIRED_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ITK_REQUIRED_LINK_FLAGS}")


#-----------------------------------------------------------------------------
# Add needed flag for gnu on linux like enviroments to build static common libs
# suitable for linking with shared object libs.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  if(NOT "${CMAKE_CXX_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
  if(NOT "${CMAKE_C_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

# END of Common.cmake

#-----------------------------------------------------------------------------
# Git protocole option
#-----------------------------------------------------------------------------
option(${CMAKE_PROJECT_NAME}_USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)
set(git_protocol "git")
if(NOT ${CMAKE_PROJECT_NAME}_USE_GIT_PROTOCOL)
  set(git_protocol "http")
endif()

find_package(Git REQUIRED)

# I don't know who removed the Find_Package for QT, but it needs to be here
# in order to build VTK if ${PRIMARY_PROJECT_NAME}_USE_QT is set.
#if(${PRIMARY_PROJECT_NAME}_USE_QT)
#find_package(Qt4 REQUIRED)
#endif()

#-----------------------------------------------------------------------------
# Enable and setup External project global properties
#-----------------------------------------------------------------------------

set(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
set(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")


include(ExternalProject)
include(ExternalProjectDependency)
include(SlicerMacroEmptyExternalProject)
include(SlicerMacroCheckExternalProjectDependency)

# Compute -G arg for configuring external projects with the same CMake generator:
if(CMAKE_EXTRA_GENERATOR)
  set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
else()
  set(gen "${CMAKE_GENERATOR}")
endif()


# With CMake 2.8.9 or later, the UPDATE_COMMAND is required for updates to occur.
# For earlier versions, we nullify the update state to prevent updates and
# undesirable rebuild.
option(FORCE_EXTERNAL_BUILDS "Force rebuilding of external project (if they are updated)" OFF)
if(CMAKE_VERSION VERSION_LESS 2.8.9 OR NOT FORCE_EXTERNAL_BUILDS)
  set(cmakeversion_external_update UPDATE_COMMAND)
  set(cmakeversion_external_update_value "" )
else()
  set(cmakeversion_external_update LOG_UPDATE )
  set(cmakeversion_external_update_value 1)
endif()

#-----------------------------------------------------------------------------
# Platform check
#-----------------------------------------------------------------------------

set(PLATFORM_CHECK true)

if(PLATFORM_CHECK)
  # See CMake/Modules/Platform/Darwin.cmake)
  #   6.x == Mac OSX 10.2 (Jaguar)
  #   7.x == Mac OSX 10.3 (Panther)
  #   8.x == Mac OSX 10.4 (Tiger)
  #   9.x == Mac OSX 10.5 (Leopard)
  #  10.x == Mac OSX 10.6 (Snow Leopard)
  if (DARWIN_MAJOR_VERSION LESS "9")
    message(FATAL_ERROR "Only Mac OSX >= 10.5 are supported !")
  endif()
endif()

#-----------------------------------------------------------------------------
# Superbuild option(s)
#-----------------------------------------------------------------------------
option(BUILD_STYLE_UTILS "Build uncrustify, cppcheck, & KWStyle" ON)
CMAKE_DEPENDENT_OPTION(
  USE_SYSTEM_Uncrustify "Use system Uncrustify program" OFF
  "BUILD_STYLE_UTILS" OFF
  )
CMAKE_DEPENDENT_OPTION(
  USE_SYSTEM_KWStyle "Use system KWStyle program" OFF
  "BUILD_STYLE_UTILS" OFF
  )
CMAKE_DEPENDENT_OPTION(
  USE_SYSTEM_Cppcheck "Use system Cppcheck program" OFF
  "BUILD_STYLE_UTILS" OFF
  )

set(EXTERNAL_PROJECT_BUILD_TYPE "Release" CACHE STRING "Default build type for support libraries")

option(USE_SYSTEM_zlib "build using the system version of zlib" OFF)
option(USE_SYSTEM_ITK "Build using an externally defined version of ITK" OFF)
option(USE_SYSTEM_SlicerExecutionModel "Build using an externally defined version of SlicerExecutionModel"  OFF)
option(USE_SYSTEM_VTK "Build using an externally defined version of VTK" OFF)
option(USE_SYSTEM_DCMTK "Build using an externally defined version of DCMTK" OFF)
option(FORCE_SYSTEM_LIBXML "Force the build using an installed version of LibXML. The building will fail if not found" OFF)

#option(${PROJECT_NAME}_BUILD_DICOM_SUPPORT "Build Dicom Support" OFF)
set(${PROJECT_NAME}_BUILD_DICOM_SUPPORT OFF)



#------------------------------------------------------------------------------
# ${PRIMARY_PROJECT_NAME} dependency list
#------------------------------------------------------------------------------
set(ITK_EXTERNAL_NAME ITKv${ITK_VERSION_MAJOR})

if (DEFINED USE_ITK_4.10)
  set(USE_ITK_4.10 ${USE_ITK_4.10} CACHE BOOL "Build using ITK 4.10 version. It may be needed in some environments because of HDF5 library incompatibilities")
else()
  set(USE_ITK_4.10 OFF CACHE BOOL "Build using ITK 4.10 version. It may be needed in some environments because of HDF5 library incompatibilities")
endif()
mark_as_superbuild(USE_ITK_4.10)

set(VTK_EXTERNAL_NAME VTKv${VTK_VERSION_MAJOR})
#if (WIN32) # libxml2 is a prerequisite for other platforms
#  set(LIBXML2_EXTERNAL_NAME LibXml2)
#else()
#  if (FORCE_SYSTEM_LIBXML)
#    find_package(LibXml2 REQUIRED)
#  else()
#    # Try first system. Otherwise use the binaries downloaded from CIPPython
#    find_package(LibXml2)
#    if (NOT LIBXML2_INCLUDE_DIR)
#      # Try to use CIPPython libraries
#      message("LIBXML libraries NOT found. Use CIPPython ones")
#      SET (LIBXML2_INCLUDE_DIR  ${CIP_PYTHON_INSTALL_DIR}/include/libxml2 CACHE PATH "")
#      SET (LIBXML2_LIBRARIES ${CIP_PYTHON_INSTALL_DIR}/lib/libxml2.dylib CACHE PATH "")
#      SET (LIBXML2_XMLLINT_EXECUTABLE ${CIP_PYTHON_INSTALL_DIR}/bin/xmllint CACHE FILEPATH "")
#    endif()
#  endif()
#endif()

#----------------
# CIP PYTHON DISTRIBUTION
#-------------------
set(CIP_PYTHON_INSTALL ON CACHE BOOL "Install Python components of CIP")
set(CIP_PYTHON_SOURCE_DIR ${CMAKE_BINARY_DIR}/CIPPython CACHE PATH "Folder where the CIP recommended Python version is downloaded" )
set(CIP_PYTHON_INSTALL_DIR ${CIP_PYTHON_SOURCE_DIR}-install CACHE PATH "Folder where the CIP recommended Python version is installed" )
if (CIP_PYTHON_INSTALL)
  if (UNIX)
    set (PYTHON_EXECUTABLE ${CIP_PYTHON_INSTALL_DIR}/bin/python2.7 CACHE FILEPATH "Python executable used for building and testing" FORCE)
  else()
    set (PYTHON_EXECUTABLE ${CIP_PYTHON_INSTALL_DIR}/python.exe CACHE FILEPATH "Python executable used for building and testing" FORCE)
  endif()
else()
  message(WARNING "CIP PYTHON will NOT be installed")
  if (NOT DEFINED PYTHON_EXECUTABLE)
    message(WARNING "Python not found. Looking for system Python...")
    FIND_PACKAGE(PythonInterp REQUIRED)  # Set the PYTHON_EXECUTABLE value
  endif()
  if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message( FATAL_ERROR "Python executable not found (${PYTHON_EXECUTABLE})" )
  endif()
  # Save the value in cache for child projects
  set (PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable" )
endif()

if (UNIX AND CIP_PYTHON_INSTALL)
  set(CIP_PYTHON_INSTALL_DL_TOOLS ON CACHE BOOL "Install Deep Learning modules of CIPPython (keras, tensorflow)")
else()
  # Deep Learning tools will not be installed in Windows
  set(CIP_PYTHON_INSTALL_DL_TOOLS OFF CACHE BOOL "Install Deep Learning modules of CIPPython (keras, tensorflow)" FORCE)
endif()

message("Using ${PYTHON_EXECUTABLE} as the active Python")


mark_as_superbuild(
 VARS
    CIP_CMAKE_CXX_FLAGS:STRING
    PYTHON_EXECUTABLE:FILEPATH
#    PYTHON_INCLUDE_DIR:PATH
#    PYTHON_LIBRARY:FILEPATH
)
#
#if (NOT DEFINED USE_BOOST)
#  # Boost will be ON by default, except for recent versions of Visual Studio compiler (>= Visual Studio 2013)
#  if (MSVC_VERSION GREATER 1700)
#    # Disable boost
#    set(USE_BOOST OFF CACHE BOOL "Enable Boost in VTK and CIP")
#    message(WARNING "Boost is not supported for Visual Studio >= 2013")
#  else()
    set(USE_BOOST ON CACHE BOOL "Enable Boost in VTK and CIP")
#  endif()
#endif()
mark_as_superbuild(USE_BOOST)

## for i in SuperBuild/*; do  echo $i |sed 's/.*External_\([a-zA-Z]*\).*/\1/g'|fgrep -v cmake|fgrep -v Template; done|sort -u
if (USE_BOOST)
  set(${PRIMARY_PROJECT_NAME}_DEPENDENCIES
    CIPPython
    SlicerExecutionModel
    ${VTK_EXTERNAL_NAME}
    ${ITK_EXTERNAL_NAME}
    Boost
    teem
    #OpenCV
  #  ${LIBXML2_EXTERNAL_NAME}
    )

else()
  set(${PRIMARY_PROJECT_NAME}_DEPENDENCIES
          CIPPython
          SlicerExecutionModel
          ${VTK_EXTERNAL_NAME}
          ${ITK_EXTERNAL_NAME}
          teem
          )
endif()

message (STATUS "PRIMARY_PROJECT_NAME_DEPENDENCIES: ${${PRIMARY_PROJECT_NAME}_DEPENDENCIES}")
#-----------------------------------------------------------------------------
# Define Superbuild global variables
#-----------------------------------------------------------------------------

# This variable will contain the list of CMake variable specific to each external project
# that should passed to ${CMAKE_PROJECT_NAME}.
# The item of this list should have the following form: <EP_VAR>:<TYPE>
# where '<EP_VAR>' is an external project variable and TYPE is either BOOL, STRING, PATH or FILEPATH.
# TODO Variable appended to this list will be automatically exported in ${PRIMARY_PROJECT_NAME}Config.cmake,
# prefix '${PRIMARY_PROJECT_NAME}_' will be prepended if it applies.
set(${CMAKE_PROJECT_NAME}_EP_VARS)

# The macro '_expand_external_project_vars' can be used to expand the list of <EP_VAR>.
set(${CMAKE_PROJECT_NAME}_EP_ARGS) # List of CMake args to configure ${PROJECT_NAME}
set(${CMAKE_PROJECT_NAME}_EP_VARNAMES) # List of CMake variable names

# Convenient macro allowing to expand the list of EP_VAR listed in ${CMAKE_PROJECT_NAME}_EP_VARS
# The expanded arguments will be appended to the list ${CMAKE_PROJECT_NAME}_EP_ARGS
# Similarly the name of the EP_VARs will be appended to the list ${CMAKE_PROJECT_NAME}_EP_VARNAMES.
macro(_expand_external_project_vars)
  set(${CMAKE_PROJECT_NAME}_EP_ARGS "")
  set(${CMAKE_PROJECT_NAME}_EP_VARNAMES "")
  foreach(arg ${${CMAKE_PROJECT_NAME}_EP_VARS})
    string(REPLACE ":" ";" varname_and_vartype ${arg})
    set(target_info_list ${target_info_list})
    list(GET varname_and_vartype 0 _varname)
    list(GET varname_and_vartype 1 _vartype)
    list(APPEND ${CMAKE_PROJECT_NAME}_EP_ARGS -D${_varname}:${_vartype}=${${_varname}})
    list(APPEND ${CMAKE_PROJECT_NAME}_EP_VARNAMES ${_varname})
  endforeach()
endmacro()

#-----------------------------------------------------------------------------
# Common external projects CMake variables
#-----------------------------------------------------------------------------
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON CACHE BOOL "Set default to prepend include directories.")

list(APPEND ${CMAKE_PROJECT_NAME}_EP_VARS
  MAKECOMMAND:STRING
  CMAKE_SKIP_RPATH:BOOL
  CMAKE_MODULE_PATH:PATH
  CMAKE_BUILD_TYPE:STRING
  BUILD_SHARED_LIBS:BOOL
  CMAKE_INCLUDE_DIRECTORIES_BEFORE:BOOL
  CMAKE_CXX_COMPILER:PATH
  CMAKE_CXX_FLAGS:STRING
  CMAKE_CXX_FLAGS_DEBUG:STRING
  CMAKE_CXX_FLAGS_MINSIZEREL:STRING
  CMAKE_CXX_FLAGS_RELEASE:STRING
  CMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING
  CMAKE_C_COMPILER:PATH
  CMAKE_C_FLAGS:STRING
  CMAKE_C_FLAGS_DEBUG:STRING
  CMAKE_C_FLAGS_MINSIZEREL:STRING
  CMAKE_C_FLAGS_RELEASE:STRING
  CMAKE_C_FLAGS_RELWITHDEBINFO:STRING
  CMAKE_EXE_LINKER_FLAGS:STRING
  CMAKE_EXE_LINKER_FLAGS_DEBUG:STRING
  CMAKE_EXE_LINKER_FLAGS_MINSIZEREL:STRING
  CMAKE_EXE_LINKER_FLAGS_RELEASE:STRING
  CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO:STRING
  CMAKE_MODULE_LINKER_FLAGS:STRING
  CMAKE_MODULE_LINKER_FLAGS_DEBUG:STRING
  CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL:STRING
  CMAKE_MODULE_LINKER_FLAGS_RELEASE:STRING
  CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO:STRING
  CMAKE_SHARED_LINKER_FLAGS:STRING
  CMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING
  CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL:STRING
  CMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING
  CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING
  CMAKE_GENERATOR:STRING
  CMAKE_EXTRA_GENERATOR:STRING
  CMAKE_INSTALL_PREFIX:PATH
  CMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH
  CMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH
  CMAKE_BUNDLE_OUTPUT_DIRECTORY:PATH
  CTEST_NEW_FORMAT:BOOL
  MEMORYCHECK_COMMAND_OPTIONS:STRING
  MEMORYCHECK_COMMAND:PATH
  CMAKE_SHARED_LINKER_FLAGS:STRING
  CMAKE_EXE_LINKER_FLAGS:STRING
  CMAKE_MODULE_LINKER_FLAGS:STRING
  SITE:STRING
  BUILDNAME:STRING
  ${PROJECT_NAME}_BUILD_DICOM_SUPPORT:BOOL
#  PYTHON_EXECUTABLE:FILEPATH
#  PYTHON_INCLUDE_DIR:PATH
#  PYTHON_LIBRARY:FILEPATH
  BUILD_EXAMPLES:BOOL
  BUILD_TESTING:BOOL
  ITK_VERSION_MAJOR:STRING
  ITK_DIR:PATH    
  #LIBXML2_INCLUDE_DIR:PATH
  #LIBXML2_LIBRARIES:PATH
  #LIBXML2_XMLLINT_EXECUTABLE:FILEPATH
  )

#if(${PRIMARY_PROJECT_NAME}_USE_QT)
#  list(APPEND ${CMAKE_PROJECT_NAME}_EP_VARS
#    ${PRIMARY_PROJECT_NAME}_USE_QT:BOOL
#    QT_QMAKE_EXECUTABLE:PATH
#    QT_MOC_EXECUTABLE:PATH
#    QT_UIC_EXECUTABLE:PATH
#    )
#endif()


_expand_external_project_vars()
set(COMMON_EXTERNAL_PROJECT_ARGS ${${CMAKE_PROJECT_NAME}_EP_ARGS})
set(extProjName ${PRIMARY_PROJECT_NAME})
set(proj        ${PRIMARY_PROJECT_NAME})
#SlicerMacroCheckExternalProjectDependency(${proj}) # JCR: This line appears to be necessary

#-----------------------------------------------------------------------------
# Set CMake OSX variable to pass down the external project
#-----------------------------------------------------------------------------
set(CMAKE_OSX_EXTERNAL_PROJECT_ARGS)
if(APPLE)
  list(APPEND CMAKE_OSX_EXTERNAL_PROJECT_ARGS
    -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
    -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

set(${PRIMARY_PROJECT_NAME}_CLI_RUNTIME_DESTINATION  bin)
set(${PRIMARY_PROJECT_NAME}_CLI_LIBRARY_DESTINATION  lib)
set(${PRIMARY_PROJECT_NAME}_CLI_ARCHIVE_DESTINATION  lib)
set(${PRIMARY_PROJECT_NAME}_CLI_INSTALL_RUNTIME_DESTINATION  bin)
set(${PRIMARY_PROJECT_NAME}_CLI_INSTALL_LIBRARY_DESTINATION  lib)
set(${PRIMARY_PROJECT_NAME}_CLI_INSTALL_ARCHIVE_DESTINATION  lib)
#-----------------------------------------------------------------------------
# Add external project CMake args
#-----------------------------------------------------------------------------
list(APPEND ${CMAKE_PROJECT_NAME}_EP_VARS
  ${PRIMARY_PROJECT_NAME}_CLI_LIBRARY_OUTPUT_DIRECTORY:PATH
  ${PRIMARY_PROJECT_NAME}_CLI_ARCHIVE_OUTPUT_DIRECTORY:PATH
  ${PRIMARY_PROJECT_NAME}_CLI_RUNTIME_OUTPUT_DIRECTORY:PATH
  ${PRIMARY_PROJECT_NAME}_CLI_INSTALL_LIBRARY_DESTINATION:PATH
  ${PRIMARY_PROJECT_NAME}_CLI_INSTALL_ARCHIVE_DESTINATION:PATH
  ${PRIMARY_PROJECT_NAME}_CLI_INSTALL_RUNTIME_DESTINATION:PATH

  INSTALL_RUNTIME_DESTINATION:STRING
  INSTALL_LIBRARY_DESTINATION:STRING
  INSTALL_ARCHIVE_DESTINATION:STRING
  )

_expand_external_project_vars()

#
# By default we want to build ${PROJECT_NAME} stuff using the CMAKE_BUILD_TYPE of
# the top level build, but build the support libraries in Release.
# So make one list of parameters to pass to ${PROJECT_NAME} when we build it and
# another for all the prerequisite libraries
#
# since we use a macro to build the list of arguments, it's easier to modify the
# list after it's built than try and conditionally change just the build type in the macro.

set(${PROJECT_NAME}_EXTERNAL_PROJECT_ARGS ${${CMAKE_PROJECT_NAME}_EP_ARGS})

set(COMMON_EXTERNAL_PROJECT_ARGS)
foreach(arg ${${CMAKE_PROJECT_NAME}_EP_ARGS})
  if(arg MATCHES "-DCMAKE_BUILD_TYPE:STRING.*")
    set(_arg -DCMAKE_BUILD_TYPE:STRING=${EXTERNAL_PROJECT_BUILD_TYPE})
  else()
    set(_arg ${arg})
  endif()
  list(APPEND COMMON_EXTERNAL_PROJECT_ARGS ${_arg})
endforeach()

#-----------------------------------------------------------------------------
set(verbose FALSE)
#-----------------------------------------------------------------------------
if(verbose)
foreach(x ${COMMON_EXTERNAL_PROJECT_ARGS})
  message("COMMON_EXTERNAL_PROJECT_ARGS:   ${x}")
endforeach()

  message("Inner external project args:")
  foreach(arg ${${CMAKE_PROJECT_NAME}_EP_ARGS})
    message("  ${arg}")
  endforeach()
endif()

string(REPLACE ";" "^" ${CMAKE_PROJECT_NAME}_EP_VARNAMES "${${CMAKE_PROJECT_NAME}_EP_VARNAMES}")

if(verbose)
  message("Inner external project argnames:")
  foreach(argname ${${CMAKE_PROJECT_NAME}_EP_VARNAMES})
    message("  ${argname}")
  endforeach()
endif()

#-----------------------------------------------------------------------------
# CTestCustom
#-----------------------------------------------------------------------------
if(BUILD_TESTING AND NOT Slicer_BUILD_${PROJECT_NAME})  
  configure_file(
    CMake/CTestCustom.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
    @ONLY)
endif()

ExternalProject_Include_Dependencies(${PRIMARY_PROJECT_NAME} DEPENDS_VAR ${PRIMARY_PROJECT_NAME}_DEPENDENCIES)

#------------------------------------------------------------------------------
# Configure and build ${PRIMARY_PROJECT_NAME}
#------------------------------------------------------------------------------
set(proj ${PRIMARY_PROJECT_NAME})
ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  DEPENDS ${${PRIMARY_PROJECT_NAME}_DEPENDENCIES}
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${PRIMARY_PROJECT_NAME}-build
  CMAKE_GENERATOR ${gen}
  #-DVTK_SOURCE_DIR:PATH=${VTK_SOURCE_DIR}
  #-DVTK_LIBRARY_DIR:PATH=${VTK_LIBRARY_DIR}
  CMAKE_ARGS
    --no-warn-unused-cli    # HACK Only expected variables should be passed down.
    ${CMAKE_OSX_EXTERNAL_PROJECT_ARGS}
    ${${PROJECT_NAME}_EXTERNAL_PROJECT_ARGS}
    -D${PRIMARY_PROJECT_NAME}_SUPERBUILD:BOOL=OFF    #NOTE: VERY IMPORTANT reprocess top level CMakeList.txt
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${CIP_CMAKE_CXX_FLAGS}
  INSTALL_COMMAND ""
  
  )

### Force rebuilding of the main subproject every time building from super structure
ExternalProject_Add_Step(${proj} forcebuild
    COMMAND ${CMAKE_COMMAND} -E remove
    ${CMAKE_CURRENT_BUILD_DIR}/${proj}-prefix/src/${proj}-stamp/${proj}-build
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
  )

