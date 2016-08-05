# TODO: add this to the automatic script
from cip_python_interfaces import *

from cip import RegisterCT, ExtractParticlesFromChestRegionChestType, LabelParticlesByChestRegionChestType, \
    ReadNRRDsWriteVTK, PerformMorphological, GetStructuresInLabelMap, EnhanceFissuresInImage, ComputeFeatureStrength, \
    RegisterLabelMaps, QualityControl, ExtractFissureBoundaryFromLobeMap, RescaleLabelMap, \
    RegionTypeLocationsToROIVolume, FilterFissureParticleData, GenerateModel, ExtractChestLabelMap, \
    FilterAirwayParticleData, EvaluateLungLobeSegmentationResults, GenerateLobeSurfaceModels, GetTransformationKappa, \
    SplitLeftLungRightLung, ComputeDistanceMap, ReadVidaWriteCIP, FindPatchMatch, ExecuteSystemCommand, \
    PerturbParticles, ConvertLabelMapValueToChestRegionChestType, GenerateBinaryThinning3D, \
    LabelMapFromRegionAndTypePoints, GenerateAtlasConvexHull, ReadWriteRegionAndTypePoints, FilterVesselParticleData, \
    ComputeCrossSectionalArea, RemoveParticlesFromParticlesDataSet, ConvertDicom, FitLobeSurfaceModelsToParticleData, \
    ComputeAirwayWallFromParticles, ComputeIntensityStatistics, GenerateRegionHistogramsAndParenchymaPhenotypes, \
    GenerateDistanceMapFromLabelMap, GetTransformationKappa2D, RemapLabelMap, MergeChestLabelMaps, \
    ClassifyFissureParticles, ReadWriteImageData, MaskOutLabelMapStructures, ConvertChestRegionChestTypeToLabelMapValue, \
    ReadVTKWriteNRRDs, GetTransformationSimilarityMetric, GenerateOtsuLungCast, CropLung, \
    RemoveChestTypeFromLabelMapUsingParticles, GenerateNLMFilteredImage, FilterConnectedComponents, ResampleLabelMap, \
    GenerateStenciledLabelMapFromParticles, ReadDicomWriteTags, ComputeFissureFeatureVectors, \
    GenerateLesionSegmentation, GenerateImageSubVolumes, RegisterLungAtlas, SegmentLungLobes, GenerateSimpleLungMask, \
    GenerateMedianFilteredImage, TransferRegionAndTypeIndicesToFromPoints, GenerateOverlayImages, MergeParticleDataSets, \
    ResampleCT, GeneratePartialLungLabelMap

