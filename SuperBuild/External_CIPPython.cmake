# Project that will add all the possible binaries to CIP
set(proj CIPPython)

# At the moment, all the binaries will be downloaded, but just one will be installed
if (UNIX)      
	if (APPLE)
		#set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-MacOSX-64.sh -f -b -p ${CIP_PYTHON_DIR})
		set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/appleScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_DIR})
	else()
		#set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-Linux-x86_64.sh -f -b -p ${CIP_PYTHON_DIR})
		set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/linuxScript.sh ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_DIR})
	endif()
else()
    # Windows
    file (TO_NATIVE_PATH ${CIP_PYTHON_DIR} CIP_PYTHON_DIR_NATIVE) # install fails without native path
	set (INSTALL_COMMAND ${CIP_PYTHON_SOURCE_DIR}/winScript.bat ${CIP_PYTHON_SOURCE_DIR} ${CIP_PYTHON_DIR_NATIVE})
endif()

# Select the master branch by default
# if (PYTHON-DEBUG-MODE)
# 	set (tag develop)
# else()
	set (tag master)
# endif()


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

# Note: pip not needed because it is installed by cython. It causes some conflicts otherwise
# ExternalProject_Add_Step(${proj} installpip
# 	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet pip  
# 	DEPENDEES install
# )

if (UNIX)      
	SET (CIP_PYTHON_BIN_DIR ${CIP_PYTHON_DIR}/bin)
else() # Windows
    SET (CIP_PYTHON_BIN_DIR ${CIP_PYTHON_DIR}/Scripts)
endif()

ExternalProject_Add_Step(${proj} installcython
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet cython  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installnumpy
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet numpy  
	DEPENDEES installcython
)

ExternalProject_Add_Step(${proj} installscipy
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet scipy  
	DEPENDEES installnumpy
)

ExternalProject_Add_Step(${proj} installvtk
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet vtk  
	DEPENDEES installscipy
)

ExternalProject_Add_Step(${proj} installpandas
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet pandas  
	DEPENDEES installvtk
)

ExternalProject_Add_Step(${proj} installnose
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet nose  
	DEPENDEES installpandas
)

ExternalProject_Add_Step(${proj} installsphinx
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet sphinx 
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installsimpleitk
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet -c SimpleITK SimpleITK 
	DEPENDEES installsphinx
)

ExternalProject_Add_Step(${proj} installlxml
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet lxml  
	DEPENDEES installsimpleitk
)

ExternalProject_Add_Step(${proj} installscikit-learn
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet scikit-learn  
	DEPENDEES installlxml
)

ExternalProject_Add_Step(${proj} installscikit-image
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet scikit-image 
	DEPENDEES installscikit-learn
)

ExternalProject_Add_Step(${proj} matplotlib
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet matplotlib 
	DEPENDEES installscikit-image
)


ExternalProject_Add_Step(${proj} installpynrrd
	COMMAND ${CIP_PYTHON_BIN_DIR}/pip install --yes --quiet pynrrd  
	DEPENDEES matplotlib
)

ExternalProject_Add_Step(${proj} installpydicom
	COMMAND ${CIP_PYTHON_BIN_DIR}/pip install --yes --quiet pydicom  
	DEPENDEES installpynrrd
)

ExternalProject_Add_Step(${proj} installnetworkx
	COMMAND ${CIP_PYTHON_BIN_DIR}/conda install --yes --quiet networkx  
	DEPENDEES installpandas
)

ExternalProject_Add_Step(${proj} installnibabel
	COMMAND ${CIP_PYTHON_BIN_DIR}/pip install --yes --quiet nibabel 
	DEPENDEES installnetworkx
)

ExternalProject_Add_Step(${proj} installnipype
	COMMAND ${CIP_PYTHON_BIN_DIR}/pip install --yes --quiet nipype
	DEPENDEES installnibabel
)




if (UNIX)      
  #Set Python variables that can be referenced by other modules
  set (CIP_PYTHON_EXECUTABLE ${CIP_PYTHON_DIR}/bin/python2.7)
  set (CIP_PYTHON_INCLUDE_DIR ${CIP_PYTHON_DIR}/include/python2.7)
  set (CIP_PYTHON_PACKAGES_PATH ${CIP_PYTHON_DIR}/lib/python2.7/site-packages)
  if (APPLE)
    set (CIP_PYTHON_LIBRARY ${CIP_PYTHON_DIR}/lib/libpython2.7.dylib)
  else()
    set (CIP_PYTHON_LIBRARY ${CIP_PYTHON_DIR}/lib/libpython2.7.so)
  endif()
else() # Windows
  #Set Python variables that can be referenced by other modules
  set (CIP_PYTHON_EXECUTABLE ${CIP_PYTHON_DIR}/python)
  set (CIP_PYTHON_INCLUDE_DIR ${CIP_PYTHON_DIR}/include)
  set (CIP_PYTHON_PACKAGES_PATH ${CIP_PYTHON_DIR}/Lib/site-packages)
  set (CIP_PYTHON_LIBRARY ${CIP_PYTHON_DIR}/python27.dll)
endif()




