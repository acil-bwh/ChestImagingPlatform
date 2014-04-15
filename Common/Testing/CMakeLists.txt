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

  ADD_TEST( NAME itkCIPAutoThresholdAirwaySegmentationImageFilterTEST COMMAND itkCIPAutoThresholdAirwaySegmentationImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD.nhdr 252 304 449 ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_interactiveAirwaySegmentation.nhdr)
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

  ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST COMMAND itkCIPOtsuLungCastImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_otsuLungCast.nhdr)
ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPPartialLungLabelMapImageFilterTEST
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

  ADD_TEST( NAME itkCIPPartialLungLabelMapImageFilterTEST COMMAND itkCIPPartialLungLabelMapImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_partialLungLabelMap.nhdr)
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

  ADD_TEST( NAME itkCIPSplitLeftLungRightLungImageFilterTEST COMMAND itkCIPSplitLeftLungRightLungImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_preLeftRightSplit.nhdr  ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_postLeftRightSplitLabeled.nhdr )
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

  ADD_TEST( NAME itkCIPLabelLungRegionsImageFilterTEST COMMAND itkCIPLabelLungRegionsImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_postLeftRightSplit.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_postLeftRightSplitLabeled.nhdr)
ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPOtsuLungCastImageFilterTEST2
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST2 COMMAND itkCIPOtsuLungCastImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/21094A_INSP_STD_NJC_COPD.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/21094A_INSP_STD_NJC_COPD_otsuLungCast.nhdr)
ENDIF( CIP_BUILD_TESTING_LARGE )
