if (NOT WIN32)
  message(FATAL_ERROR "External project definition of LibXml2 is valid only on Win32.")
endif()
  
set(proj LibXml2)

# Set dependency list
set(${proj}_DEPENDENCIES "")

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

#if(${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})
#  unset(LibXml2_DIR CACHE)
#  find_package(LIBXML2 REQUIRED)
#  set(LIBXML2_INCLUDE_DIR ${LIBXML2_INCLUDE_DIRS})
#  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES})
#endif()

# Sanity checks
if(DEFINED LibXml2_DIR AND NOT EXISTS ${LibXml2_DIR})
  message(FATAL_ERROR "LibXml2_DIR variable is defined but corresponds to non-existing directory")
endif()

if(NOT DEFINED LibXml2_DIR AND NOT ${CMAKE_PROJECT_NAME}_USE_SYSTEM_${proj})

  if(NOT DEFINED git_protocol)
    set(git_protocol "git")
  endif()

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}/win32)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)
  file(TO_NATIVE_PATH ${EP_INSTALL_DIR} EP_NATIVE_INSTALL_DIR)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "${git_protocol}://git.gnome.org/libxml2" # warning: very slow
    GIT_TAG "v2.9.1"
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CONFIGURE_COMMAND cscript configure.js iconv=no prefix=${EP_NATIVE_INSTALL_DIR}
    BUILD_COMMAND nmake /f Makefile.msvc
    INSTALL_COMMAND nmake install
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(LibXml2_DIR ${EP_INSTALL_DIR})
  set(LIBXML2_ROOT ${LibXml2_DIR})
  set(LIBXML2_INCLUDE_DIR ${LibXml2_DIR}/include/libxml2)
  set(LIBXML2_LIBRARIES ${LibXml2_DIR}/lib/libxml2.lib)
else()
  # The project is provided using LibXml2_DIR, nevertheless since other project may depend on LibXml2,
  # let's add an 'empty' one
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

mark_as_superbuild(
  VARS
    LIBXML2_INCLUDE_DIR:PATH
    LIBXML2_LIBRARIES:FILEPATH
    LIBXML2_ROOT:PATH
  LABELS "FIND_PACKAGE"
  )

ExternalProject_Message(${proj} "LIBXML2_INCLUDE_DIR:${LIBXML2_INCLUDE_DIR}")
ExternalProject_Message(${proj} "LIBXML2_LIBRARIES:${LIBXML2_LIBRARIES}")
if(LIBXML2_ROOT)
  ExternalProject_Message(${proj} "LIBXML2_ROOT:${LIBXML2_ROOT}")
endif()
