CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT( CIPUtilities )

# --------------------------------------------------------------------------
# Set INCLUDE_DIRS variable
# --------------------------------------------------------------------------
set(${PROJECT_NAME}_INCLUDE_DIRS 
       ${CMAKE_CURRENT_SOURCE_DIR}/VTK 
       ${CMAKE_CURRENT_BINARY_DIR}/VTK
       ${CMAKE_CURRENT_SOURCE_DIR}/ITK 
       ${CMAKE_CURRENT_BINARY_DIR}/ITK
     CACHE INTERNAL "${PROJECT_NAME} include dirs" FORCE)


include_directories(${CIPUtilities_INCLUDE_DIRS})

# --------------------------------------------------------------------------
# Configure headers
# --------------------------------------------------------------------------
set(configure_header_file vtkCIPUtilitiesConfigure.h)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/VTK/${configure_header_file}.in
  ${CMAKE_CURRENT_BINARY_DIR}/VTK/${configure_header_file}
  )

# --------------------------------------------------------------------------
# Install headers
# --------------------------------------------------------------------------
if(NOT DEFINED ${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT)
  set(${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT ON)
endif()
if(NOT ${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT)
  file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.(h|txx)")
  install(
    FILES ${headers} ${CMAKE_CURRENT_BINARY_DIR}/${configure_header_file}
    DESTINATION include/${PROJECT_NAME} COMPONENT Development)
endif()


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
SET (CIP_UTILITIES_VTK ${CMAKE_SOURCE_DIR}/Utilities/VTK)

SET ( CIPUtilities_SRCS
  ${CIP_UTILITIES_VTK}/vtkSuperquadricTensorGlyphFilter.cxx
  ${CIP_UTILITIES_VTK}/vtkGlyph3DWithScaling.cxx
  ${CIP_UTILITIES_VTK}/vtkImageConnectivity.cxx           
  ${CIP_UTILITIES_VTK}/vtkImageErode.cxx
  ${CIP_UTILITIES_VTK}/vtkImageNeighborhoodFilter.cxx
  ${CIP_UTILITIES_VTK}/vtkNRRDReader.cxx
  ${CIP_UTILITIES_VTK}/vtkNRRDWriter.cxx
)

# --------------------------------------------------------------------------
# Build library
# --------------------------------------------------------------------------

set( LIB_NAME ${PROJECT_NAME} )

add_library(${LIB_NAME} 
         ${CIPUtilities_SRCS} 
  )

TARGET_LINK_LIBRARIES ( ${LIB_NAME} ${VTK_LIBRARIES} ${Teem_LIBRARIES} )

# --------------------------------------------------------------------------
# Export target
# --------------------------------------------------------------------------
if(NOT DEFINED ${PROJECT_NAME}_EXPORT_FILE)
  set(${PROJECT_NAME}_EXPORT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake)
endif()
export(TARGETS ${lib_name} APPEND FILE ${${PROJECT_NAME}_EXPORT_FILE})

# --------------------------------------------------------------------------
# Install library
# --------------------------------------------------------------------------
if(NOT DEFINED ${PROJECT_NAME}_INSTALL_BIN_DIR)
  set(${PROJECT_NAME}_INSTALL_BIN_DIR bin)
endif()
if(NOT DEFINED ${PROJECT_NAME}_INSTALL_LIB_DIR)
  set(${PROJECT_NAME}_INSTALL_LIB_DIR lib/${PROJECT_NAME})
endif()

install(TARGETS ${lib_name}
  RUNTIME DESTINATION ${${PROJECT_NAME}_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
  LIBRARY DESTINATION ${${PROJECT_NAME}_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${${PROJECT_NAME}_INSTALL_LIB_DIR} COMPONENT Development
  )

# --------------------------------------------------------------------------
# Python Wrapping
# --------------------------------------------------------------------------
#if(VTK_WRAP_PYTHON)
#  include(vtkMacroKitPythonWrap)

#  vtkMacroKitPythonWrap(
#    KIT_NAME ${lib_name}
#    KIT_SRCS ${CIPUtilities_SRCS}
#    KIT_INSTALL_BIN_DIR ${${PROJECT_NAME}_INSTALL_BIN_DIR}
#    KIT_INSTALL_LIB_DIR ${${PROJECT_NAME}_INSTALL_LIB_DIR}
#    )
  # Export target
#  export(TARGETS ${lib_name}Python ${lib_name}PythonD APPEND FILE ${${PROJECT_NAME}_EXPORT_FILE})
#endif()




