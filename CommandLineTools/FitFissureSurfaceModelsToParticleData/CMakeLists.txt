CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FitFissureSurfaceModelsToParticleData )

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

SET ( VTK_LIBS
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering 
vtkWidgets 
vtkHybrid
)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( FitFissureSurfaceModelsToParticleData_SOURCE FitFissureSurfaceModelsToParticleData.cxx )

ADD_EXECUTABLE ( FitFissureSurfaceModelsToParticleData ${FitFissureSurfaceModelsToParticleData_SOURCE})
TARGET_LINK_LIBRARIES ( FitFissureSurfaceModelsToParticleData ${ITK_LIBRARIES} ${VTK_LIBS}  CIPIO CIPCommon )
