CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( MergeParticleDataSets )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( MergeParticleDataSets_SOURCE MergeParticleDataSets.cxx )

ADD_EXECUTABLE ( MergeParticleDataSets ${MergeParticleDataSets_SOURCE}  )
TARGET_LINK_LIBRARIES ( MergeParticleDataSets ${VTK_LIBRARIES} ${ITK_LIBRARIES} )



