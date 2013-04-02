CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ACILAssistant )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

FIND_PACKAGE( FLTK )
IF ( FLTK_FOUND )
  INCLUDE_DIRECTORIES (${FLTK_INCLUDE_DIR})
ELSE ( FLTK_FOUND )
  MESSAGE( FATAL_ERROR "This application requires FLTK.")
ENDIF ( FLTK_FOUND )

SET(InsightApplications_SOURCE_DIR "Folder Undefined" CACHE PATH "Set InsightApplications_SOURCE_DIR")
SET(InsightApplications_BINARY_DIR "Folder Undefined" CACHE PATH "Set InsightApplications_BINARY_DIR")

IF ( ${InsightApplications_BINARY_DIR} STREQUAL "Folder Undefined" )
  MESSAGE( FATAL_ERROR "Must set InsightApplications_SOURCE_DIR and InsightApplications_BINARY_DIR.")
ENDIF ( ${InsightApplications_BINARY_DIR} STREQUAL "Folder Undefined" )

IF ( ${InsightApplications_SOURCE_DIR} STREQUAL "Folder Undefined" )
  MESSAGE( FATAL_ERROR "Must set InsightApplications_SOURCE_DIR.")
ENDIF ( ${InsightApplications_SOURCE_DIR} STREQUAL "Folder Undefined" )

INCLUDE_DIRECTORIES(
${InsightApplications_SOURCE_DIR}/Auxiliary/FltkImageViewer
${InsightApplications_BINARY_DIR}/Auxiliary/FltkImageViewer
${InsightApplications_SOURCE_DIR}/Auxiliary/vtk
${InsightApplications_BINARY_DIR}/Auxiliary/vtk
${InsightApplications_SOURCE_DIR}
${InsightApplications_BINARY_DIR}
${InsightApplications_SOURCE_DIR}/Auxiliary/VtkFltk   
${CMAKE_SOURCE_DIR}/Common
${CMAKE_SOURCE_DIR}/IO
)

LINK_DIRECTORIES ( 
${InsightApplications_BINARY_DIR}/VolviewPlugIns/bin/
${InsightApplications_BINARY_DIR}/VolviewPlugIns/bin/
)

SET ( ACILAssistant_SOURCE ACILAssistant.cxx PaintBrushAndEraserGUI.cxx ACILAssistantBase.cxx )

SET ( VTK_LIBS
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering
)

ADD_EXECUTABLE ( ACILAssistant ${ACILAssistant_SOURCE})
TARGET_LINK_LIBRARIES ( ACILAssistant ITKFltkImageViewer ITKVtkFltk ${ITK_LIBRARIES} ${VTK_LIBS} ${FLTK_LIBRARIES} CIPCommon CIPIO )
