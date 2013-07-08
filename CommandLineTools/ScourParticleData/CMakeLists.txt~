CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( ScourParticleData )

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )  
   INCLUDE ( ${VTK_USE_FILE} )    
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF( VTK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( VTK_LIBS
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering 
vtkWidgets 
vtkHybrid
)

SET ( ScourParticleData_SOURCE ScourParticleData.cxx )

ADD_EXECUTABLE ( ScourParticleData ${ScourParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( ScourParticleData ${VTK_LIBS} )
