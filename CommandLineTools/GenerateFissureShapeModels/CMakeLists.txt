CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( GenerateFissureShapeModels )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/IO )

SET ( GenerateFissureShapeModels_SOURCE GenerateFissureShapeModels.cxx )

ADD_EXECUTABLE ( GenerateFissureShapeModels ${GenerateFissureShapeModels_SOURCE})
TARGET_LINK_LIBRARIES ( GenerateFissureShapeModels ${ITK_LIBRARIES} ${VTK_LIBRARIES} CIPIO )



