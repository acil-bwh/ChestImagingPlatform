ADD_EXECUTABLE( ConvertDicomTEST ConvertDicomTEST.cxx)
TARGET_LINK_LIBRARIES( ConvertDicomTEST )

SET_TARGET_PROPERTIES ( ConvertDicomTEST
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/CommandLineTools/ConvertDicom/Testing"
)

ADD_TEST( ConvertDicomTEST ConvertDicomTEST )
