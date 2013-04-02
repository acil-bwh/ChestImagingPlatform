CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LabelParticlesByChestRegionChestType )

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
vtkIO 
vtkCommon 
vtkImaging 
vtkFiltering 
vtkGraphics 
vtkRendering 
vtkWidgets 
vtkHybrid
)

SET ( LabelParticlesByChestRegionChestType_SOURCE LabelParticlesByChestRegionChestType.cxx )

ADD_EXECUTABLE ( LabelParticlesByChestRegionChestType ${LabelParticlesByChestRegionChestType_SOURCE} )
TARGET_LINK_LIBRARIES ( LabelParticlesByChestRegionChestType ${VTK_LIBS} ${ITK_LIBRARIES} )
