CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RemoveParticlesFromParticlesDataSet )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

SET ( VTK_LIBS
vtkCommon
vtkIO
)

SET ( RemoveParticlesFromParticlesDataSet_SOURCE RemoveParticlesFromParticlesDataSet.cxx )

ADD_EXECUTABLE ( RemoveParticlesFromParticlesDataSet ${RemoveParticlesFromParticlesDataSet_SOURCE}  )
TARGET_LINK_LIBRARIES ( RemoveParticlesFromParticlesDataSet ${VTK_LIBS} )



