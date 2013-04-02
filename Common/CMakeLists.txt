CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CIPCommon )

INCLUDE_DIRECTORIES( 
${CMAKE_SOURCE_DIR}/Utilities/ITK
${CMAKE_SOURCE_DIR}/Utilities/VTK
)

FIND_PACKAGE( VTK )
IF ( VTK_FOUND )
  INCLUDE (${VTK_USE_FILE})
ELSE ( VTK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without VTK" )
ENDIF (VTK_FOUND)

FIND_PACKAGE( ITK )
IF ( ITK_FOUND )
  INCLUDE (${ITK_USE_FILE})
ELSE ( ITK_FOUND )
  MESSAGE ( FATAL_ERROR "Cannot build without ITK" )
ENDIF (ITK_FOUND)

FIND_PACKAGE( Teem )
IF ( Teem_FOUND )
   INCLUDE (${Teem_USE_FILE})
ELSE ( Teem_FOUND )
   MESSAGE ( FATAL_ERROR "Cannot build without Teem" )
ENDIF (Teem_FOUND)        

SET ( CIPCommon_SRCS
  vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.cxx
  vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.cxx
  #cipConnectedAirwayParticlesToHMMAirwayGraphFunctor.cxx  
  cipThinPlateSplineSurface.cxx
  cipLobeBoundaryShapeModel.cxx
  cipChestRegionChestTypeLocations.cxx
  cipSphereStencil.cxx
  cipCylinderStencil.cxx
  cipChestDataViewer.cxx
  cipAirwayDataInteractor.cxx
  cipVesselDataInteractor.cxx
  itkCIPExtractChestLabelMapImageFilter.cxx
  cipParticleConnectedComponentFilter.cxx
  cipVesselParticleConnectedComponentFilter.cxx
  cipAirwayParticleConnectedComponentFilter.cxx
  cipFissureParticleConnectedComponentFilter.cxx
  cipThinPlateSplineSurfaceModelToParticlesMetric.cxx
  cipNelderMeadSimplexOptimizer.cxx
  cipParticleToThinPlateSplineSurfaceMetric.cxx
  cipHelper.cxx
  )

IF(VTK_USE_BOOST)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
  SET ( CIPCommon_SRCS 
        ${CIPCommon_SRCS}
	    vtkCIPAirwayParticlesToGenerationLabeledAirwayParticlesFilter.cxx
        vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.cxx
       )
ENDIF (VTK_USE_BOOST)
       
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

