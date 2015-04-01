# Project that will add all the possible binaries to CIP
set(proj CIPPython)

# At the moment, all the binaries will be downloaded, but just one will be installed
if (UNIX)      
	if (APPLE)
		set (INSTALL_COMMAND bash ${CIP_PYTHON_DIR}/Miniconda-MacOSX-64.sh -f -b -p ${CIP_PYTHON_DIR}-install)
	else()
		set (INSTALL_COMMAND bash ${CIP_PYTHON_DIR}/Miniconda-Win-x86_64.exe -f -b -p ${CIP_PYTHON_DIR}-install)
	endif()
else()
      # Windows
	set (INSTALL_COMMAND ${CIP_PYTHON_DIR}/Miniconda-3.8.3-Windows-x86_64.exe /InstallationType=AllUsers /S /D=${CIP_PYTHON_DIR}-install)
endif()

# Select the master branch by default
if (PYTHON-DEBUG-MODE)
	set (tag develop)
else()
	set (tag master)
endif()

ExternalProject_Add(${proj}
	GIT_REPOSITORY "${git_protocol}://github.com/acil-bwh/CIPPython.git"
	GIT_TAG ${tag}
	SOURCE_DIR ${CIP_PYTHON_DIR}     
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ${INSTALL_COMMAND}
)

ExternalProject_Add_Step(${proj} installnumpy
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet numpy  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installscipy
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet scipy  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installvtk
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet vtk  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installpandas
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet pandas  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installcython
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet cython  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installnose
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet nose  
	DEPENDEES install
)

ExternalProject_Add_Step(${proj} installpynrrd
	COMMAND ${CIP_PYTHON_DIR}-install/bin/conda install --yes --quiet pynrrd  
	DEPENDEES install
)