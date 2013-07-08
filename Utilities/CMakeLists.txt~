CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CIPUtilities )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

SET ( CIPUtilities_SRCS
  ${CMAKE_SOURCE_DIR}/Utilities/VTK/vtkSuperquadricTensorGlyphFilter.cxx
)

set( LIB_NAME CIPUtilities )

add_library(${LIB_NAME}
  ${CIPUtilities_SRCS} 
  )

SET ( VTK_LIBS
vtkCommon 
)

TARGET_LINK_LIBRARIES ( ${LIB_NAME} ${VTK_LIBS} )