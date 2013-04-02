CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(TransferFieldDataToFromPointData )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( TransferFieldDataToFromPointData_SOURCE TransferFieldDataToFromPointData.cxx )

ADD_EXECUTABLE (TransferFieldDataToFromPointData ${TransferFieldDataToFromPointData_SOURCE})
TARGET_LINK_LIBRARIES ( TransferFieldDataToFromPointData ${VTK_LIBRARIES} ${ITK_LIBRARIES} )
