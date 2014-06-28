# Make sure that the ExtProjName/IntProjName variables are unique globally
# even if other External_${ExtProjName}.cmake files are sourced by
# SlicerMacroCheckExternalProjectDependency
set(extProjName BOOST) #The find_package known name
set(proj        Boost) #This local name
set(${extProjName}_REQUIRED_VERSION "")  #If a required version is necessary, then set this, else leave blank

# Sanity checks
if(DEFINED ${extProjName}_DIR AND NOT EXISTS ${${extProjName}_DIR})
  message(FATAL_ERROR "${extProjName}_DIR variable is defined but corresponds to non-existing directory (${${extProjName}_DIR})")
endif()

if(NOT ( DEFINED "USE_SYSTEM_${extProjName}" AND "${USE_SYSTEM_${extProjName}}" ) )

  set(EXTERNAL_PROJECT_OPTIONAL_ARGS)

  set(CMAKE_PROJECT_INCLUDE_EXTERNAL_PROJECT_ARG)

  ### --- Project specific additions here
  set(Boost_Install_Dir ${CMAKE_CURRENT_BINARY_DIR}/${proj}-install)
  set(Boost_Configure_Script ${CMAKE_CURRENT_LIST_DIR}/External_Boost_configureboost.cmake)
  set(Boost_Build_Script ${CMAKE_CURRENT_LIST_DIR}/External_Boost_buildboost.cmake)

  ### --- End Project specific additions
# SVN is too slow SVN_REPOSITORY http://svn.boost.org/svn/boost/trunk
# SVN is too slow SVN_REVISION -r "82586"
  set(${proj}_URL http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz )
  set(${proj}_MD5 93780777cfbf999a600f62883bd54b17 )
  if(CMAKE_COMPILER_IS_CLANGXX)
    set(CLANG_ARG -DCMAKE_COMPILER_IS_CLANGXX:BOOL=ON)
  endif()
  ExternalProject_Add(${proj}
    URL ${${proj}_URL}
    URL_MD5 ${${proj}_MD5}
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}
    ${cmakeversion_external_update} "${cmakeversion_external_update_value}"
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
    ${CLANG_ARG}
    -DBUILD_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/${proj}
    -DBOOST_INSTALL_DIR:PATH=${Boost_Install_Dir}
    -P ${Boost_Configure_Script}
	-P ${Boost_Build_Script}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND}
    INSTALL_COMMAND ""
  )
  set(BOOST_ROOT        ${Boost_Install_Dir})
  set(BOOST_INCLUDE_DIR ${Boost_Install_Dir}/include)
else()
  if(${USE_SYSTEM_${extProjName}})
    find_package(${proj} ${${extProjName}_REQUIRED_VERSION} REQUIRED)
    message("USING the system ${extProjName}, set ${extProjName}_DIR=${${extProjName}_DIR}")
  endif()
  # The project is provided using ${extProjName}_DIR, nevertheless since other
  # project may depend on ${extProjName}, let's add an 'empty' one
  SlicerMacroEmptyExternalProject(${proj} "${${proj}_DEPENDENCIES}")
endif()

mark_as_superbuild(
  VARS BOOST_DIR:PATH
  LABELS "FIND_PACKAGE"
  )
