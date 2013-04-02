CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterAirwayParticleData )

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )  
   INCLUDE ( ${VTK_USE_FILE} )    
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF( VTK_FOUND )

FIND_PACKAGE ( ITK )
IF ( ITK_FOUND )
  INCLUDE(${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF ( ITK_FOUND )

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

SET ( FilterAirwayParticleData_SOURCE FilterAirwayParticleData.cxx )

ADD_EXECUTABLE ( FilterAirwayParticleData ${FilterAirwayParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FilterAirwayParticleData ${ITK_LIBRARIES} ${VTK_LIBS} CIPCommon )
