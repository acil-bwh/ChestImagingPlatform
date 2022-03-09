cmake_minimum_required(VERSION 3.5.0)
cmake_policy(VERSION 3.5.0)

set(PRIMARY_PROJECT_NAME CIP)

project(${PRIMARY_PROJECT_NAME})


#-----------------------------------------------------------------------------
# Set C++ standard version and extensions
#-----------------------------------------------------------------------------
# Cache the variable values to allow setting a value externally.
set(_minimum_cxx_standard "11")
set(CMAKE_CXX_STANDARD "${_minimum_cxx_standard}" CACHE STRING "C++ standard")
if("${CMAKE_CXX_STANDARD}" LESS "${_minimum_cxx_standard}")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD must be equal or larger than ${_minimum_cxx_standard}")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE STRING "C++ standard required")
set(CMAKE_CXX_EXTENSIONS OFF CACHE STRING "C++ extensions")
message(STATUS "${_msg} - C++${CMAKE_CXX_STANDARD}")

#-----------------------------------------------------------------------------
# Superbuild Option - Enabled by default
#                   Phase I:  ${PRIMARY_PROJECT_NAME}_SUPERBUILD is set to ON, and the
#                             supporting packages defined in "SuperBuild.cmake"
#                             are built.  The last package in "SuperBuild.cmake"
#                             to be built is a recursive call to this
#                             file with ${PRIMARY_PROJECT_NAME}_SUPERBUILD explicitly
#                             set to "OFF" to initiate Phase II
#
#                   Phase II: Build the ${PRIMARY_PROJECT_NAME}, referencing the support
#                             packages built in Phase I.
#-----------------------------------------------------------------------------
option(${PRIMARY_PROJECT_NAME}_SUPERBUILD "Build ${PRIMARY_PROJECT_NAME} and the projects it depends on via SuperBuild.cmake." ON)
#mark_as_advanced(${PRIMARY_PROJECT_NAME}_SUPERBUILD)

#-----------------------------------------------------------------------------
# Superbuild script
#-----------------------------------------------------------------------------
message("----------- ENTRY ------------")

if(${PRIMARY_PROJECT_NAME}_SUPERBUILD)
  message("Configuring Superbuild...")
  #project(SuperBuild_${PRIMARY_PROJECT_NAME})  # note the change of project name
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  message("----------- DONE ------------")
  return()
endif()

message("----------- PHASE 2 ------------")
message("Configuring ${PRIMARY_PROJECT_NAME}...")
include("${CMAKE_CURRENT_SOURCE_DIR}/${PRIMARY_PROJECT_NAME}.cmake")
message("----------- DONE ------------")
