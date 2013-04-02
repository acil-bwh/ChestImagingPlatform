CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RegisterLungAtlas )

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

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( VTK_LIBS
vtkIO 
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering 
vtkWidgets 
vtkHybrid
)

SET ( RegisterLungAtlas_SOURCE RegisterLungAtlas.cxx )

ADD_EXECUTABLE ( RegisterLungAtlas ${RegisterLungAtlas_SOURCE} )
TARGET_LINK_LIBRARIES ( RegisterLungAtlas ${VTK_LIBS} ${ITK_LIBRARIES} )
