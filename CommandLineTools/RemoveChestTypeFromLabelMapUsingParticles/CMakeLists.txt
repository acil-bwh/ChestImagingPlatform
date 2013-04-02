CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RemoveChestTypeFromLabelMapUsingParticles )

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

SET ( VTK_LIBS
vtkCommon
vtkIO
)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( RemoveChestTypeFromLabelMapUsingParticles_SOURCE RemoveChestTypeFromLabelMapUsingParticles.cxx )

ADD_EXECUTABLE ( RemoveChestTypeFromLabelMapUsingParticles ${RemoveChestTypeFromLabelMapUsingParticles_SOURCE} )
TARGET_LINK_LIBRARIES ( RemoveChestTypeFromLabelMapUsingParticles ${VTK_LIBS} ${ITK_LIBRARIES} )
