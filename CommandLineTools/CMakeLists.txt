CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT( CommandLineTools )

IF (WIN32)
ADD_DEFINITIONS(-DUSE_STATIC_CIP_LIBS)
ENDIF (WIN32)

SET(CLIS_DIRS
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
)

#Append CLIS that depends on BOOST
IF (CIP_USE_BOOST)

  list (APPEND ${CLIS_DIRS}
    ReadParticlesWriteConnectedParticles
    LabelAirwayParticlesByGeneration
    GenerateStatisticsForAirwayGenerationLabeling
  )

ENDIF(CIP_USE_BOOST)

foreach (CLI ${CLIS_DIRS})
  set(BUILD_${CLI} ON CACHE BOOL "BUILD_${CLI}" )
  if(BUILD_${CLI})
      SUBDIRS (${CLI})
  endif(BUILD_${CLI})
endforeach(CLI)



SET(BUILD_LEGACYCLIS OFF CACHE BOOL "BUILD_LEGACYCLIS")
IF(BUILD_LEGACYCLIS)
  SUBDIRS (LegacyCLIs)
ENDIF(BUILD_LEGACYCLIS)



