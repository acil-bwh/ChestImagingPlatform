PROJECT ( CIP_PYTHON )

CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/setup_ChestConventions.py.in 
  ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py @ONLY )

ADD_CUSTOM_TARGET ( ${PROJECT_NAME}
  COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/setup_ChestConventions.py build_ext -b ${CIP_BINARY_DIR}/bin
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
ADD_DEPENDENCIES ( ChestConventions ${PROJECT_NAME} )

IF ( CIP_BUILD_TESTING )
 SUBDIRS ( particles )
 SUBDIRS ( phenotypes )
ENDIF( CIP_BUILD_TESTING ) 
