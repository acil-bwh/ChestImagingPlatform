CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateModel )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

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

SET ( GenerateModel_SOURCE GenerateModel.cxx )

SET ( VTK_LIBS
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering 
vtkWidgets 
vtkHybrid
)

ADD_EXECUTABLE ( GenerateModel ${GenerateModel_SOURCE})
TARGET_LINK_LIBRARIES ( GenerateModel ${ITK_LIBRARIES} ${VTK_LIBS} )



