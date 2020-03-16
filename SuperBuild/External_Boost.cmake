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
  if (UNIX)
    set(Boost_Install_Dir ${CMAKE_CURRENT_BINARY_DIR}/${proj}-install)
  else()
    set(Boost_Install_Dir ${CMAKE_CURRENT_BINARY_DIR}/${proj})
  endif()
  set(Boost_Configure_Script ${CMAKE_CURRENT_LIST_DIR}/External_Boost_configureboost.cmake)
  set(Boost_Build_Script ${CMAKE_CURRENT_LIST_DIR}/External_Boost_buildboost.cmake)

  ### --- End Project specific additions
# SVN is too slow SVN_REPOSITORY http://svn.boost.org/svn/boost/trunk
# SVN is too slow SVN_REVISION -r "82586"

  #set(${proj}_URL http://sourceforge.net/projects/boost/files/boost/1.54.0/boost_1_54_0.tar.gz )
#  set(${proj}_URL https://acil.s3.amazonaws.com/external_deps/boost_1_54_0_nodoc.tar.gz)
#  set(${proj}_MD5 81bb79d6939601b43e681449e3eae7df )

#    set(${proj}_URL https://s3.amazonaws.com/acil/external_deps/boost_1_65_1.tar.gz)
#    set(${proj}_MD5 ee64fd29a3fe42232c6ac3c419e523cf )
    
    set(${proj}_URL https://s3.amazonaws.com/acil/external_deps/boost_1_72_0.tar.gz)
    set(${proj}_MD5 e2b0b1eac302880461bcbef097171758 )

  if(CMAKE_COMPILER_IS_CLANGXX)
    set(CLANG_ARG -DCMAKE_COMPILER_IS_CLANGXX:BOOL=ON)
  endif()
  ExternalProject_Add(${proj}
    URL ${${proj}_URL}
    URL_MD5 ${${proj}_MD5}

    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}
#     URL    /Users/jonieva/Projects/External/boost_1_65_1.tar.gz
#    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E echo "Remove this line and uncomment GIT_REPOSITORY and GIT_TAG"
#    SOURCE_DIR /Users/jonieva/Projects/External/boost_1_65_1

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

  set(BOOST_ROOT ${Boost_Install_Dir})

else()
  if(${USE_SYSTEM_${extProjName}})
    find_package(${proj} ${${extProjName}_REQUIRED_VERSION} REQUIRED)
    message("USING the system ${extProjName}, set ${extProjName}_DIR=${${extProjName}_DIR}")
  endif()
  # The project is provided using ${extProjName}_DIR, nevertheless since other
  # project may depend on ${extProjName}, let's add an 'empty' one
  SlicerMacroEmptyExternalProject(${proj} "${${proj}_DEPENDENCIES}")
endif()

mark_as_superbuild(BOOST_ROOT)