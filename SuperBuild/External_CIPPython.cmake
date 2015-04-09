# Project that will add all the possible binaries to CIP
set(proj CIPPython)

# At the moment, all the binaries will be downloaded, but just one will be installed
if (UNIX)      
	if (APPLE)
		set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-MacOSX-64.sh -f -b -p ${CIP_PYTHON_DIR})
	else()
		set (INSTALL_COMMAND bash ${CIP_PYTHON_SOURCE_DIR}/Miniconda-Linux-x86_64.sh -f -b -p ${CIP_PYTHON_DIR})
	endif()
else()
      # Windows
	set (INSTALL_COMMAND ${CIP_PYTHON_SOURCE_DIR}/Miniconda-3.8.3-Windows-x86_64.exe /InstallationType=AllUsers /S /D=${CIP_PYTHON_DIR})
endif()

# Select the master branch by default
if (PYTHON-DEBUG-MODE)
	set (tag develop)
else()
	set (tag master)
endif()

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

ExternalProject_Add_Step(${proj} installcython
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet cython  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installnumpy
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet numpy  
	DEPENDEES installcython
)

ExternalProject_Add_Step(${proj} installscipy
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet scipy  
	DEPENDEES installnumpy
)

ExternalProject_Add_Step(${proj} installvtk
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet vtk  
	DEPENDEES installscipy
)

ExternalProject_Add_Step(${proj} installpandas
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet pandas  
	DEPENDEES installvtk
)

ExternalProject_Add_Step(${proj} installnose
	COMMAND ${CIP_PYTHON_DIR}/bin/conda install --yes --quiet nose  
	DEPENDEES installpandas
)

ExternalProject_Add_Step(${proj} installpynrrd
	COMMAND ${CIP_PYTHON_DIR}/bin/pip install --quiet pynrrd  
	DEPENDEES installnose
)

ExternalProject_Add_Step(${proj} installpydicom
	COMMAND ${CIP_PYTHON_DIR}/bin/pip install --quiet pydicom  
	DEPENDEES installpynrrd
)




