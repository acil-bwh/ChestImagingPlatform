# Project that will add all the possible binaries to CIP
set(proj CIPPython)

set(CIPPython ${CMAKE_BINARY_DIR}/CIPPython)


# At the moment, all the binaries will be downloaded, but just one will be installed
if (UNIX)      
	if (APPLE)
		set (INSTALL_COMMAND bash ${CIPPython}/Miniconda-MacOSX-64.sh -f -b -p ${CIPPython}-install)
	else()
		set (INSTALL_COMMAND bash ${CIPPython}/Miniconda-Win-x86_64.exe -f -b -p ${CIPPython}-install)
	endif()
else()
      # Windows
	set (INSTALL_COMMAND ${CIPPython}/Miniconda-3.8.3-Windows-x86_64.exe /InstallationType=AllUsers /S /D=${CIPPython}-install)
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
	SOURCE_DIR ${CIPPython}     
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ${INSTALL_COMMAND}
)

ExternalProject_Add_Step(${proj} installNumpy
	COMMAND ${CIPPython}-install/bin/conda install --yes --quiet numpy  
	DEPENDEES install
)
