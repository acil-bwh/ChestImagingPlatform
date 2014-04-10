CMAKE_MINIMUM_REQUIRED(VERSION 2.8.4)

PROJECT( CIPCommon )


# --------------------------------------------------------------------------
# Include dirs
# --------------------------------------------------------------------------
set(${PROJECT_NAME}_INCLUDE_DIRS 
      ${CIPUtilities_INCLUDE_DIRS} 
      ${CMAKE_CURRENT_BINARY_DIR}
     CACHE INTERNAL "${PROJECT_NAME} include dirs" FORCE)

INCLUDE_DIRECTORIES( ${CIPCommon_INCLUDE_DIRS}  )  

# --------------------------------------------------------------------------
# Configure headers
# --------------------------------------------------------------------------
set(configure_header_file vtkCIPCommonConfigure.h)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/${configure_header_file}.in
  ${CMAKE_CURRENT_BINARY_DIR}/${configure_header_file}
  )

# --------------------------------------------------------------------------
# Install headers
# --------------------------------------------------------------------------
if(NOT DEFINED ${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT)
  set(${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT ON)
endif()
if(NOT ${PROJECT_NAME}_INSTALL_NO_DEVELOPMENT)
  file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.(h|txx)")
  install(
    FILES ${headers} ${CMAKE_CURRENT_BINARY_DIR}/${configure_header_file}
    DESTINATION include/${PROJECT_NAME} COMPONENT Development)
endif()

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
SET ( CIPCommon_SRCS
  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.cxx
  vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.cxx
  #cipConnectedAirwayParticlesToHMMAirwayGraphFunctor.cxx  
  cipThinPlateSplineSurface.cxx
  cipLobeSurfaceModel.cxx
  cipChestRegionChestTypeLocations.cxx
  cipSphereStencil.cxx
  cipCylinderStencil.cxx
  cipChestDataViewer.cxx
  cipAirwayDataInteractor.cxx
  cipVesselDataInteractor.cxx
  itkCIPExtractChestLabelMapImageFilter.cxx
  itkCIPMergeChestLabelMapsImageFilter.cxx
  cipParticleConnectedComponentFilter.cxx
  cipVesselParticleConnectedComponentFilter.cxx
  cipAirwayParticleConnectedComponentFilter.cxx
  cipFissureParticleConnectedComponentFilter.cxx
  cipLeftLobesThinPlateSplineSurfaceModelToParticlesMetric.cxx
  cipRightLobesThinPlateSplineSurfaceModelToParticlesMetric.cxx
  cipThinPlateSplineSurfaceModelToParticlesMetric.cxx
  cipNelderMeadSimplexOptimizer.cxx
  cipParticleToThinPlateSplineSurfaceMetric.cxx
  cipHelper.cxx
  vtkSimpleLungMask.cxx
  vtkImageStatistics.cxx
  cipExceptionObject.cxx
  cipChestConventions.cxx
  )

IF(VTK_USE_BOOST)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
  SET ( CIPCommon_SRCS 
        ${CIPCommon_SRCS}
	vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.cxx
        vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.cxx
       )
ENDIF (VTK_USE_BOOST)
       
# --------------------------------------------------------------------------
# Build library
# --------------------------------------------------------------------------
set( LIB_NAME CIPCommon )

add_library(${LIB_NAME}
  ${CIPCommon_SRCS} 
  )

target_link_libraries(${LIB_NAME}
  ${ITK_LIBRARIES}
  ${VTK_LIBRARIES}
  CIPUtilities
#  ${Teem_LIBRARIES}
  )

IF ( CIP_BUILD_TESTING )
 SUBDIRS ( Testing )
ENDIF( CIP_BUILD_TESTING ) 
