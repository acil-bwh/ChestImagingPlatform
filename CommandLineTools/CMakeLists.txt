CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CommandLineTools )

if (WIN32)
  ADD_DEFINITIONS(-DUSE_STATIC_CIP_LIBS)
endif (WIN32)

set(CLIS_DIRS
  ExampleCLI
  ConvertDicom
  MergeParticleDataSets
  RemoveParticlesFromParticlesDataSet
  ExtractParticlesFromChestRegionChestType
  GenerateOtsuLungCast
  GenerateAtlasConvexHull
  RegisterLungAtlas
  ResampleLabelMap
  RescaleLabelMap
  ResampleCT
  QualityControl
  RemoveChestTypeFromLabelMapUsingParticles
  SegmentLungLobes
  ReadDicomWriteTags
  SplitLeftLungRightLung
  ExtractChestLabelMap
  LabelParticlesByChestRegionChestType
  GenerateStenciledLabelMapFromParticles
  GenerateDistanceMapFromLabelMap
  FilterVesselParticleData
  FilterAirwayParticleData
  ReadNRRDsWriteVTK
  ReadVTKWriteNRRDs
  ComputeCrossSectionalArea
  ComputeIntensityStatistics
  FilterFissureParticleData
  GenerateLobeSurfaceModels
  ClassifyFissureParticles
  CropLung
  GenerateModel
  GenerateImageSubVolumes
  GenerateMedianFilteredImage
  FindPatchMatch
  TransferRegionAndTypeIndicesToFromPoints
  EvaluateLungLobeSegmentationResults
  ConvertLabelMapValueToChestRegionChestType
  GenerateParenchymaPhenotypes
  ComputeDistanceMap
  PerturbParticles
  ReadWriteRegionAndTypePoints
  PerformMorphological
  GenerateOverlayImages
  GetTransformationSimilarityMetric
  GetTransformationKappa2D
  GetTransformationKappa
  ReadWriteImageData
  GetStructuresInLabelMap
  GeneratePartialLungLabelMap
  ReadVidaWriteCIP
  RegisterLabelMaps
  RegisterCT
  MergeChestLabelMaps
  MaskOutLabelMapStructures
  ComputeFeatureStrength
  LabelMapFromRegionAndTypePoints
  GenerateSimpleLungMask
  GenerateBinaryThinning3D
  ConvertChestRegionChestTypeToLabelMapValue
  FilterConnectedComponents
  EnhanceFissuresInImage
  ComputeFissureFeatureVectors
  FitLobeSurfaceModelsToParticleData
  GenerateLesionSegmentation
  RemapLabelMap
  GenerateNLMFilteredImage
  ComputeAirwayWallFromParticles
  RegionTypeLocationsToROIVolume
  ExtractFissureBoundaryFromLobeMap
  ExecuteSystemCommand
  GraphCutsOptimization
  SegmentLungAirways
  ComputeStrainFromDeformationField
  ComputeAirwayWallFromImages
  FitBSplineToDeformationField
  CreateVolumeFromSlices
)

#Append CLIS that depends on BOOST
if ( CIP_USE_BOOST )
  message ( "Appending new list" )
  list (APPEND CLIS_DIRS
    ReadParticlesWriteConnectedParticles
    LabelAirwayParticlesByGeneration
    GenerateStatisticsForAirwayGenerationLabeling
  )

endif( )

foreach ( CLI ${CLIS_DIRS} )
  set(BUILD_${CLI} ON CACHE BOOL "BUILD_${CLI}" )
  if(BUILD_${CLI})
      SUBDIRS (${CLI})
  endif(BUILD_${CLI})
endforeach( )



set(BUILD_LEGACYCLIS ON CACHE BOOL "BUILD_LEGACYCLIS")

if ( BUILD_LEGACYCLIS )
  SUBDIRS (LegacyCLIs)
endif( )



