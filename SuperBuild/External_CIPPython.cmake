# Project that will add all the possible binaries to CIP
set(proj CIPPython)

SET(INSTALL_CIP_PYTHON_DISTRIBUTION ON CACHE BOOL "INSTALL_CIP_PYTHON_DISTRIBUTION")
mark_as_superbuild(INSTALL_CIP_PYTHON_DISTRIBUTION)

SET(CIP_PYTHON_USE_QT4 OFF CACHE BOOL "Use Qt4 in CIP Python (it can be used in case of VTK errors)")
mark_as_superbuild(CIP_PYTHON_USE_QT4)

if (INSTALL_CIP_PYTHON_DISTRIBUTION)
  # At the moment, all the binaries will be downloaded, but just one will be installed
  if (UNIX)
    if (APPLE)
      #set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-MacOSX-64.sh -f -b -p ${CIP_PYTHON_INSTALL_DIR})
      set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/appleScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR})
    else()
      #set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-Linux-x86_64.sh -f -b -p ${CIP_PYTHON_INSTALL_DIR})
      set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/linuxScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR})
    endif()
  else()
      # Windows
      file (TO_NATIVE_PATH ${CIP_PYTHON_INSTALL_DIR} CIP_PYTHON_INSTALL_DIR_NATIVE) # install fails without native path
    set (INSTALL_COMMAND ${CIP_PYTHON_SOURCE_DIR}/winScript.bat ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_INSTALL_DIR_NATIVE})
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

  if (UNIX)
    SET (CIP_PYTHON_BIN_DIR ${CIP_PYTHON_INSTALL_DIR}/bin)
  else() # Windows
      SET (CIP_PYTHON_BIN_DIR ${CIP_PYTHON_INSTALL_DIR}/Scripts)
  endif()

  #### Conda-forge packages
  if (UNIX)
    # Tensorflow 1.2.1 not available in Windows for Python 2
    ExternalProject_Add_Step(${proj} installtensorflow
        COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c conda-forge tensorflow==1.2.1
        #DEPENDEES installscikit-image
        DEPENDEES install
    )
    ExternalProject_Add_Step(${proj} installkeras
        COMMAND ${CIP_PYTHON_BIN_DIR}/pip install keras==2.0.8
        DEPENDEES installtensorflow
    )
    SET (last_dep installkeras)
  else()
    SET (last_dep install)
  endif()

  ExternalProject_Add_Step(${proj} installnipype
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c conda-forge nipype==0.12.1
          DEPENDEES ${last_dep}
          )

  ExternalProject_Add_Step(${proj} installnetworkx
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c conda-forge networkx==1.11
          DEPENDEES installnipype
  )

  ExternalProject_Add_Step(${proj} installscikit-image
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes -c conda-forge scikit-image
          DEPENDEES installnetworkx
  )


  #### pip packages
  ExternalProject_Add_Step(${proj} installpynrrd
          COMMAND ${CIP_PYTHON_BIN_DIR}/pip install pynrrd
          DEPENDEES installscikit-image
          )

  ExternalProject_Add_Step(${proj} installpydicom
          COMMAND ${CIP_PYTHON_BIN_DIR}/pip install pydicom==0.9.9
          DEPENDEES installpynrrd
          )

  ExternalProject_Add_Step(${proj} installnibabel
          COMMAND ${CIP_PYTHON_BIN_DIR}/pip install nibabel
          DEPENDEES installpydicom
          )

  #### Conda packages
  ExternalProject_Add_Step(${proj} installnumpy
          COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes numpy
          DEPENDEES installnibabel
  )

  ExternalProject_Add_Step(${proj} installcython
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes cython
	DEPENDEES installnumpy
  )

  ExternalProject_Add_Step(${proj} installscipy
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes scipy
    DEPENDEES installcython
  )

  ExternalProject_Add_Step(${proj} installvtk
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes vtk
    DEPENDEES installscipy
  )

  ExternalProject_Add_Step(${proj} installpandas
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes pandas
    DEPENDEES installvtk
  )

  ExternalProject_Add_Step(${proj} installnose
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes nose
    DEPENDEES installpandas
  )

  ExternalProject_Add_Step(${proj} installsphinx
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes sphinx
    DEPENDEES installnose
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

  ExternalProject_Add_Step(${proj} installxml
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes lxml
    DEPENDEES installsimpleitk
  )

  ExternalProject_Add_Step(${proj} installscikit-learn
    COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes scikit-learn
    DEPENDEES installxml
  )

  if (CIP_PYTHON_USE_QT4)
    # Force qt 4.8.7 (to reuse for VTK build)
    ExternalProject_Add_Step(${proj} installqt4
            COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes qt=4.8.7
            DEPENDEES installscikit-learn
            )
  endif()

else()
  # Ignore CIPPython
  ExternalProject_Add_Empty(${proj})
endif()



