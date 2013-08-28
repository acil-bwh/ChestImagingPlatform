CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( MergeAndFilterRecons )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Utilities )

SET ( MergeAndFilterRecons_SOURCE MergeAndFilterRecons.cxx )

ADD_EXECUTABLE ( MergeAndFilterRecons ${MergeAndFilterRecons_SOURCE} )
TARGET_LINK_LIBRARIES ( MergeAndFilterRecons ${ITK_LIBRARIES}  )
