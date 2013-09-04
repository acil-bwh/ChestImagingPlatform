CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CIPUtilities )

SET ( CIPUtilities_SRCS
  ${CMAKE_SOURCE_DIR}/Utilities/VTK/vtkSuperquadricTensorGlyphFilter.cxx
  ${CMAKE_SOURCE_DIR}/Utilities/VTK/vtkGlyph3DWithScaling.cxx

)

set( LIB_NAME CIPUtilities )

add_library(${LIB_NAME}
  ${CIPUtilities_SRCS} 
  )


TARGET_LINK_LIBRARIES ( ${LIB_NAME} ${VTK_LIBRARIES} )
