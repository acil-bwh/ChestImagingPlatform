CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( RemoveParticlesFromParticlesDataSet )
SET ( RemoveParticlesFromParticlesDataSet_SOURCE RemoveParticlesFromParticlesDataSet.cxx )

ADD_EXECUTABLE ( RemoveParticlesFromParticlesDataSet ${RemoveParticlesFromParticlesDataSet_SOURCE}  )
TARGET_LINK_LIBRARIES ( RemoveParticlesFromParticlesDataSet ${VTK_LIBRARIES} )



