# IMPORTANT: CIP testing depends on MIDAS server to download the data files

  
# Login in MIDAS and save the token in midas_api_token
# If there is a problem with the token, token = 0, and it won't be used
midas_api_login(
    MIDAS_REST_URL ${MIDAS_REST_URL}
    MIDAS_USER_EMAIL ${MIDAS_USER_EMAIL}
    MIDAS_USER_APIKEY ${MIDAS_USER_APIKEY}
    RESULT_VARNAME token
)
message("Token (result from login): ${token}")
#SET (token "LlXLKgvHtNvWBLFiu7D5Y8gDCq3iMJDzQD7wLyqO")
# SET (token "0")

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

  #ADD_TEST( NAME itkCIPAutoThresholdAirwaySegmentationImageFilterTEST COMMAND itkCIPAutoThresholdAirwaySegmentationImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000.nhdr 252 304 449 ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_interactiveAirwaySegmentation.nhdr)
  midas_add_test( NAME itkCIPAutoThresholdAirwaySegmentationImageFilterTEST COMMAND itkCIPAutoThresholdAirwaySegmentationImageFilterTEST MIDAS_TOKEN{${token}} MIDAS{Case000.nhdr.md5} 252 304 449 MIDAS{Case000_interactiveAirwaySegmentation.nhdr.md5} EXTRAFILES MIDAS{Case000.raw.gz.md5} MIDAS{Case000_interactiveAirwaySegmentation.raw.gz.md5})
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

  ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST COMMAND itkCIPOtsuLungCastImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_otsuLungCast.nhdr)
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

  ADD_TEST( NAME itkCIPPartialLungLabelMapImageFilterTEST COMMAND itkCIPPartialLungLabelMapImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_partialLungLabelMap.nhdr)
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

  ADD_TEST( NAME itkCIPSplitLeftLungRightLungImageFilterTEST COMMAND itkCIPSplitLeftLungRightLungImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_preLeftRightSplit.nhdr  ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_postLeftRightSplitLabeled.nhdr )
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

  ADD_TEST( NAME itkCIPLabelLungRegionsImageFilterTEST COMMAND itkCIPLabelLungRegionsImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_postLeftRightSplit.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/Case000_postLeftRightSplitLabeled.nhdr)
ENDIF( CIP_BUILD_TESTING_LARGE )

#----------------------------------------------
# itkCIPOtsuLungCastImageFilterTEST2
#----------------------------------------------
IF ( CIP_BUILD_TESTING_LARGE )
  ADD_TEST( NAME itkCIPOtsuLungCastImageFilterTEST2 COMMAND itkCIPOtsuLungCastImageFilterTEST ${CIP_SOURCE_DIR}/Testing/Data/Large/Case001.nhdr ${CIP_SOURCE_DIR}/Testing/Data/Large/Case001_otsuLungCast.nhdr)
ENDIF( CIP_BUILD_TESTING_LARGE )
