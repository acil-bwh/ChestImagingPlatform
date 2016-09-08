CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( LegacyCLIs )

IF (WIN32)
ADD_DEFINITIONS(-DUSE_STATIC_CIP_LIBS)
ENDIF (WIN32)

SET(CLIS_DIRS
  TransferFieldDataToFromPointData
  UpdateParticlesData
)


foreach (CLI ${CLIS_DIRS})
  set(BUILD_${CLI} ON CACHE BOOL "BUILD_${CLI}" )
  if(BUILD_${CLI})
     SUBDIRS (${CLI})
  endif(BUILD_${CLI})
endforeach(CLI)



