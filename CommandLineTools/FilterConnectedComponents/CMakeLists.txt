CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( FilterConnectedComponents )

SET ( MODULE_NAME FilterConnectedComponents )

ADD_LIBRARY( FilterConnectedComponentsHelper FilterConnectedComponentsHelper.cxx )

TARGET_LINK_LIBRARIES( FilterConnectedComponentsHelper
  ${ITK_LIBRARIES} 
  CIPCommon
  CIPIO
)

SET ( MODULE_TARGET_LIBRARIES
 FilterConnectedComponentsHelper
  )

SET ( MODULE_SRCS FilterConnectedComponents.cxx )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

IF ( CIP_BUILD_TESTING )
  SUBDIRS ( Testing )
ENDIF( CIP_BUILD_TESTING ) 

