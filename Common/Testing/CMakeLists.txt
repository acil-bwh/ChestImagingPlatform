# IMPORTANT: CIP Large testing depends on MIDAS server to download the data files


#----------------------------------------------
# itkCIPAutoThresholdAirwaySegmentationImageFilterTEST
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  PROJECT ( itkCIPAutoThresholdAirwaySegmentationImageFilterTEST )	

  INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

  ADD_EXECUTABLE( itkCIPAutoThresholdAirwaySegmentationImageFilterTEST itkCIPAutoThresholdAirwaySegmentationImageFilterTEST.cxx )
  TARGET_LINK_LIBRARIES( itkCIPAutoThresholdAirwaySegmentationImageFilterTEST CIPCommon )

  SET_TARGET_PROPERTIES ( itkCIPAutoThresholdAirwaySegmentationImageFilterTEST 
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
  )

  MIDAS_ADD_TEST( NAME itkCIPAutoThresholdAirwaySegmentationImageFilterTEST COMMAND itkCIPAutoThresholdAirwaySegmentationImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case000.nhdr.md5} 252 304 449 MIDAS{Case000_interactiveAirwaySegmentation.nhdr.md5} EXTRAFILES MIDAS{Case000.raw.gz.md5} MIDAS{Case000_interactiveAirwaySegmentation.raw.gz.md5})

ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPOtsuLungCastImageFilterTEST
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  PROJECT ( itkCIPOtsuLungCastImageFilterTEST )	

  INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

  ADD_EXECUTABLE( itkCIPOtsuLungCastImageFilterTEST itkCIPOtsuLungCastImageFilterTEST.cxx )
  TARGET_LINK_LIBRARIES( itkCIPOtsuLungCastImageFilterTEST CIPCommon )

  SET_TARGET_PROPERTIES ( itkCIPOtsuLungCastImageFilterTEST 
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
  )

  MIDAS_ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST COMMAND itkCIPOtsuLungCastImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case000.nhdr.md5} MIDAS{Case000_otsuLungCast.nhdr.md5} EXTRAFILES MIDAS{Case000.raw.gz.md5} MIDAS{Case000_otsuLungCast.raw.gz.md5} )
  
ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPPartialLungLabelMapImageFilterTEST
# This test is specially heavy, it takes around 25 minutes to run in a powerful machine
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  PROJECT ( itkCIPPartialLungLabelMapImageFilterTEST )	

  INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

  ADD_EXECUTABLE( itkCIPPartialLungLabelMapImageFilterTEST itkCIPPartialLungLabelMapImageFilterTEST.cxx )
  TARGET_LINK_LIBRARIES( itkCIPPartialLungLabelMapImageFilterTEST CIPCommon )

  SET_TARGET_PROPERTIES ( itkCIPPartialLungLabelMapImageFilterTEST 
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
  )

  MIDAS_ADD_TEST( NAME itkCIPPartialLungLabelMapImageFilterTEST COMMAND itkCIPPartialLungLabelMapImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case000.nhdr.md5} MIDAS{Case000_partialLungLabelMap.nhdr.md5} EXTRAFILES MIDAS{Case000.raw.gz.md5} MIDAS{Case000_partialLungLabelMap.raw.gz.md5} )
ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPSplitLeftLungRightLungImageFilterTEST
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  PROJECT ( itkCIPSplitLeftLungRightLungImageFilterTEST )	

  INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

  ADD_EXECUTABLE( itkCIPSplitLeftLungRightLungImageFilterTEST itkCIPSplitLeftLungRightLungImageFilterTEST.cxx )
  TARGET_LINK_LIBRARIES( itkCIPSplitLeftLungRightLungImageFilterTEST CIPCommon )

  SET_TARGET_PROPERTIES ( itkCIPSplitLeftLungRightLungImageFilterTEST 
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
  )

  MIDAS_ADD_TEST( NAME itkCIPSplitLeftLungRightLungImageFilterTEST COMMAND itkCIPSplitLeftLungRightLungImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case000.nhdr.md5} MIDAS{Case000_preLeftRightSplit.nhdr.md5} MIDAS{Case000_postLeftRightSplitLabeled.nhdr.md5} EXTRAFILES MIDAS{Case000.raw.gz.md5} MIDAS{Case000_preLeftRightSplit.raw.gz.md5} MIDAS{Case000_postLeftRightSplitLabeled.raw.gz.md5} )

ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPLabelLungRegionsImageFilterTEST
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  PROJECT ( itkCIPLabelLungRegionsImageFilterTEST )	

  INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

  ADD_EXECUTABLE( itkCIPLabelLungRegionsImageFilterTEST itkCIPLabelLungRegionsImageFilterTEST.cxx )
  TARGET_LINK_LIBRARIES( itkCIPLabelLungRegionsImageFilterTEST CIPCommon )

  SET_TARGET_PROPERTIES ( itkCIPLabelLungRegionsImageFilterTEST 
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
  )

  MIDAS_ADD_TEST( NAME itkCIPLabelLungRegionsImageFilterTEST COMMAND itkCIPLabelLungRegionsImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case000_postLeftRightSplit.nhdr.md5} MIDAS{Case000_postLeftRightSplitLabeled.nhdr.md5} EXTRAFILES MIDAS{Case000_postLeftRightSplit.raw.gz.md5} MIDAS{Case000_postLeftRightSplitLabeled.raw.gz.md5})

ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPOtsuLungCastImageFilterTEST2
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  MIDAS_ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST2 COMMAND itkCIPOtsuLungCastImageFilterTEST MIDAS_TOKEN{${MIDAS_AUTH_TOKEN}} MIDAS{Case001.nhdr.md5} MIDAS{Case001_otsuLungCast.nhdr.md5} EXTRAFILES MIDAS{Case001.raw.gz.md5} MIDAS{Case001_otsuLungCast.raw.gz.md5} )
ENDIF( CIP_BUILD_TESTING_LARGE )

#-----------------------------------
# itkCIPExtractChestLabelMapImageFilterTEST
#-----------------------------------
PROJECT ( itkCIPExtractChestLabelMapImageFilterTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

ADD_EXECUTABLE( itkCIPExtractChestLabelMapImageFilterTEST itkCIPExtractChestLabelMapImageFilterTEST.cxx)
TARGET_LINK_LIBRARIES( itkCIPExtractChestLabelMapImageFilterTEST CIPCommon )

SET_TARGET_PROPERTIES ( itkCIPExtractChestLabelMapImageFilterTEST
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
)

ADD_TEST( itkCIPExtractChestLabelMapImageFilterTEST itkCIPExtractChestLabelMapImageFilterTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/simple_lm.nrrd )

#-----------------------------------
# itkCIPMergeChestLabelMapsImageFilterTEST
#-----------------------------------
PROJECT ( itkCIPMergeChestLabelMapsImageFilterTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

ADD_EXECUTABLE( itkCIPMergeChestLabelMapsImageFilterTEST itkCIPMergeChestLabelMapsImageFilterTEST.cxx)
TARGET_LINK_LIBRARIES( itkCIPMergeChestLabelMapsImageFilterTEST CIPCommon )

SET_TARGET_PROPERTIES ( itkCIPMergeChestLabelMapsImageFilterTEST 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
)

ADD_TEST( itkCIPMergeChestLabelMapsImageFilterTEST itkCIPMergeChestLabelMapsImageFilterTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/simple_lm.nrrd )

#-----------------------------------
# cipHelperTEST
#-----------------------------------
PROJECT ( cipHelperTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

ADD_EXECUTABLE( cipHelperTEST cipHelperTEST.cxx)
TARGET_LINK_LIBRARIES( cipHelperTEST CIPCommon )

SET_TARGET_PROPERTIES ( cipHelperTEST 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
)

ADD_TEST( cipHelperTEST cipHelperTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/simple_lm.nrrd )

#-----------------------------------
# cipNelderMeadSimplexOptimizerTEST
#-----------------------------------
PROJECT ( cipNelderMeadSimplexOptimizerTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

ADD_EXECUTABLE( cipNelderMeadSimplexOptimizerTEST cipNelderMeadSimplexOptimizerTEST.cxx)
TARGET_LINK_LIBRARIES( cipNelderMeadSimplexOptimizerTEST CIPCommon )

SET_TARGET_PROPERTIES ( cipNelderMeadSimplexOptimizerTEST 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
)

ADD_TEST( cipNelderMeadSimplexOptimizerTEST cipNelderMeadSimplexOptimizerTEST )

#-----------------------------------
# cipLobeSurfaceModelTEST
#-----------------------------------
PROJECT ( cipLobeSurfaceModelTEST )

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/Common )

ADD_EXECUTABLE( cipLobeSurfaceModelTEST cipLobeSurfaceModelTEST.cxx)
TARGET_LINK_LIBRARIES( cipLobeSurfaceModelTEST CIPCommon CIPIO )

SET_TARGET_PROPERTIES ( cipLobeSurfaceModelTEST 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CIP_BINARY_DIR}/Common/Testing"
)

ADD_TEST( cipLobeSurfaceModelTEST cipLobeSurfaceModelTEST ${CMAKE_SOURCE_DIR}/Testing/Data/Input/Case000_rightLungLobesShapeModel.csv )
