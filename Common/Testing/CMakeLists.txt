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

  ADD_TEST( NAME itkCIPAutoThresholdAirwaySegmentationImageFilterTEST COMMAND itkCIPAutoThresholdAirwaySegmentationImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/18501Q_INSP_STD_BWH_COPD_interactiveAirwaySegmentation.nhdr)
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
