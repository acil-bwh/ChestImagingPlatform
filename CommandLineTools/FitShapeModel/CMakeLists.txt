cmake_minimum_required(VERSION 2.8.0)

project(FitShapeModel)

set(MODULE_NAME FitShapeModel)
set(MODULE_SRCS)

ADD_LIBRARY( ShapeModelHelper
             ShapeModel.cxx
             ShapeModelInitializer.cxx
             ShapeModelOptimizer.cxx
             ShapeModelOptimizerCT.cxx
             ShapeModelOptimizerDT.cxx
             ShapeModelFinalizer.cxx
             ShapeModelVisualizer.cxx
             ShapeModelImageFactory.cxx
             VNLVTKConverters.cxx
             ShapeModelUtils.cxx
             cnpy.cpp )

TARGET_LINK_LIBRARIES( ShapeModelHelper
  ${VTK_LIBRARIES}
  ${ITK_LIBRARIES}
  PoissonRecon
)

SUBDIRS( PoissonRecon )

find_package(LibXml2)

set (UUID_LIBS -luuid)

set(MODULE_INCLUDE_DIRECTORIES
  ${LIBXML2_INCLUDE_DIR}
  )

set(MODULE_TARGET_LIBRARIES
  ${LIBXML2_LIBRARIES}
  ShapeModelHelper
  )

cipMacroBuildCLI(
    NAME ${MODULE_NAME}
    ADDITIONAL_TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
    ADDITIONAL_INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
    SRCS ${MODULE_SRCS}
    )

SET (TEST_NAME ${MODULE_NAME}_Test)
CIP_ADD_TEST(NAME ${TEST_NAME} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
    --compare
      ${INPUT_DATA_DIR}/lm-64-transformed.nrrd
      ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
    ModuleEntryPoint
      -l ${INPUT_DATA_DIR}/lm-64.nrrd
      -d ${INPUT_DATA_DIR}/lm-64-transformed.nrrd
      -t ${INPUT_DATA_DIR}/TranslationRotation.tfm
      -m 3
      -r ${OUTPUT_DATA_DIR}/${TEST_NAME}_dummy_lm.nrrd
)
