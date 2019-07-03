# Project that will add all the possible binaries to CIP
set(proj CIPPython)

SET(CIP_PYTHON_USE_QT4 OFF CACHE BOOL "Use Qt4 in CIP Python (it can be used in case of VTK errors)")
mark_as_superbuild(CIP_PYTHON_USE_QT4)

if (CIP_PYTHON_INSTALL)
  # At the moment, all the binaries will be downloaded, but just one will be installed
  if (UNIX)
    if (APPLE)
      set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/appleScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR})
    else()
      set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/linuxScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR})
    endif()
  else()
      # Windows
      file (TO_NATIVE_PATH ${CIP_PYTHON_INSTALL_DIR} CIP_PYTHON_INSTALL_DIR_WINDOWS) # install fails without native path
      set (INSTALL_COMMAND ${CIP_PYTHON_SOURCE_DIR}/winScript.bat ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR_WINDOWS})
  endif()

  # Select the master branch by default
  set (tag master)

  # Install Miniconda
  ExternalProject_Add(${proj}
    GIT_REPOSITORY "${git_protocol}://github.com/acil-bwh/CIPPython.git"
    GIT_TAG ${tag}
    SOURCE_DIR ${CIP_PYTHON_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${INSTALL_COMMAND}
  )

  # Install Python packages.
  # Every package depends on the previous one to allow multi-threading in cmake. Otherwise conda will make trouble when installing packages in parallel

  # Get the folder where pip and conda executables are installed
  get_filename_component(CIP_PYTHON_BIN_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
  if (WIN32)
    set(CIP_PYTHON_BIN_DIR ${CIP_PYTHON_BIN_DIR}/Scripts)
  endif()



    ExternalProject_Add_Step(${proj} installnipype
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c conda-forge nipype
          DEPENDEES install
          )


    ExternalProject_Add_Step(${proj} installscikit-image
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes scikit-image
          DEPENDEES installnipype
          )

    if (UNIX)
      ExternalProject_Add_Step(${proj} installsimpleitk
        COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c SimpleITK SimpleITK
        DEPENDEES installsphinx
      )
    else()
      # Unknown conflict with SimpleITK 1.1.0. For the time being, force 0.9.1
      ExternalProject_Add_Step(${proj} installsimpleitk
              COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c SimpleITK SimpleITK=0.9.1
              DEPENDEES installsphinx
      )
    endif()

    ExternalProject_Add_Step(${proj} installgitpython
      COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes gitpython
      DEPENDEES installscikit-learn
    )

    if (UNIX)
        ExternalProject_Add_Step(${proj} installvtk
                COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes vtk
                DEPENDEES installscipy
                )
    else()
        # VTK 8.X not compatible with Python 2.7 in Windows. Install VTK 7.1.1 from a custom build (the "official" ones don't work)
        ExternalProject_Add_Step(${proj} installvtk
                COMMAND ${CIP_PYTHON_BIN_DIR}/pip install ${CIP_PYTHON_SOURCE_DIR}/VTK-7.1.1-cp27-cp27m-win_amd64.whl
                DEPENDEES installscipy
                )
    endif()


  ExternalProject_Add_Step(${proj} installpytables
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes pytables
    DEPENDEES installgitpython
  )

  ExternalProject_Add_Step(${proj} installpandas
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes pandas
    DEPENDEES installvtk
  )

  ExternalProject_Add_Step(${proj} installpynrrd
          COMMAND ${CIP_PYTHON_BIN_DIR}/pip install pynrrd
          DEPENDEES installscikit-image
          )

  ExternalProject_Add_Step(${proj} installscikit-learn
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes scikit-learn
    DEPENDEES installxml
  )

  ExternalProject_Add_Step(${proj} installcython
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes cython
	DEPENDEES installnumpy
  )

  ExternalProject_Add_Step(${proj} installsphinx
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes sphinx
    DEPENDEES installnose
  )


  ########################################################################################
  #### Deep Learning dependencies
  ########################################################################################
  if (CIP_PYTHON_INSTALL_DL_TOOLS)
    message("Python Deep Learning modules (tensorflow) will be installed")
    ExternalProject_Add_Step(${proj} installtensorflow
            COMMAND ${CIP_PYTHON_BIN_DIR}/pip install tensorflow==1.12.0
            DEPENDEES installgitpython
    )
    SET (last_dep installtensorflow)
  else()
    message("Python Deep Learning modules will NOT be installed")
    SET (last_dep installpytables)
  endif()
else()
  # Ignore CIPPython
  ExternalProject_Add_Empty(${proj})
endif()



