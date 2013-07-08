CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( MergeParticleDataSets )

FIND_PACKAGE ( VTK )
IF ( VTK_FOUND )
  INCLUDE(${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF ( VTK_FOUND )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( MergeParticleDataSets_SOURCE MergeParticleDataSets.cxx )

ADD_EXECUTABLE ( MergeParticleDataSets ${MergeParticleDataSets_SOURCE}  )
TARGET_LINK_LIBRARIES ( MergeParticleDataSets ${VTK_LIBRARIES} )



