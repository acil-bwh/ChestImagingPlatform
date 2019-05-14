import os
import collections
import numpy as np
import time
import subprocess
import operator
import math
import vtk
import csv
import itertools
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from optparse import OptionParser
from cip_python.segmentation import NoduleSegmenter
from cip_python.common import ChestConventions

class FirstOrderStatistics:
    def __init__(self, parameterValues, bins, grayLevels, allKeys):
        """
        :param parameterValues: 3D array with the coordinates of the voxels where the labelmap is not 0
        :param bins: bins for histogram
        :param grayLevels: number of different gray levels
        :param allKeys: all feature keys that have been selected for analysis
        """
        self.firstOrderStatistics = collections.OrderedDict()
        self.firstOrderStatisticsTiming = collections.OrderedDict()
        self.firstOrderStatistics["Voxel Count"] = "self.voxelCount(self.parameterValues)"
        self.firstOrderStatistics["Gray Levels"] = "self.grayLevelCount(self.grayLevels)"
        self.firstOrderStatistics["Energy"] = "self.energyValue(self.parameterValues)"
        self.firstOrderStatistics["Entropy"] = "self.entropyValue(self.bins)"
        self.firstOrderStatistics["Minimum Intensity"] = "self.minIntensity(self.parameterValues)"
        self.firstOrderStatistics["Maximum Intensity"] = "self.maxIntensity(self.parameterValues)"
        self.firstOrderStatistics["Mean Intensity"] = "self.meanIntensity(self.parameterValues)"
        self.firstOrderStatistics["Median Intensity"] = "self.medianIntensity(self.parameterValues)"
        self.firstOrderStatistics["Range"] = "self.rangeIntensity(self.parameterValues)"
        self.firstOrderStatistics["Mean Deviation"] = "self.meanDeviation(self.parameterValues)"
        self.firstOrderStatistics["Root Mean Square"] = "self.rootMeanSquared(self.parameterValues)"
        self.firstOrderStatistics["Standard Deviation"] = "self.standardDeviation(self.parameterValues)"
        self.firstOrderStatistics["Ventilation Heterogeneity"] = "self.ventilationHeterogeneity(self.parameterValues)"
        self.firstOrderStatistics["Skewness"] = "self.skewnessValue(self.parameterValues)"
        self.firstOrderStatistics["Kurtosis"] = "self.kurtosisValue(self.parameterValues)"
        self.firstOrderStatistics["Variance"] = "self.varianceValue(self.parameterValues)"
        self.firstOrderStatistics["Uniformity"] = "self.uniformityValue(self.bins)"
        self.firstOrderStatistics["LAA950Perc"] = "self.laa950PercValue(self.parameterValues)"
        self.firstOrderStatistics["LAA910Perc"] = "self.laa910PercValue(self.parameterValues)"
        self.firstOrderStatistics["LAA856Perc"] = "self.laa856PercValue(self.parameterValues)"

        self.parameterValues = parameterValues
        self.bins = bins
        self.grayLevels = grayLevels
        self.keys = set(allKeys).intersection(self.firstOrderStatistics.keys())

    def voxelCount(self, parameterArray):
        return (parameterArray.size)

    def grayLevelCount(self, grayLevels):
        return (grayLevels)

    def energyValue(self, parameterArray):
        return (np.sum(parameterArray ** 2))

    def entropyValue(self, bins):
        return (np.sum(bins * np.where(bins != 0, np.log2(bins), 0)))

    def minIntensity(self, parameterArray):
        return (np.min(parameterArray))

    def maxIntensity(self, parameterArray):
        return (np.max(parameterArray))

    def meanIntensity(self, parameterArray):
        return (np.mean(parameterArray))

    def medianIntensity(self, parameterArray):
        return (np.median(parameterArray))

    def rangeIntensity(self, parameterArray):
        return (np.max(parameterArray) - np.min(parameterArray))

    def meanDeviation(self, parameterArray):
        return (np.mean(np.absolute((np.mean(parameterArray) - parameterArray))))

    def rootMeanSquared(self, parameterArray):
        return (((np.sum(parameterArray ** 2)) / (parameterArray.size)) ** (1 / 2.0))

    def standardDeviation(self, parameterArray):
        return (np.std(parameterArray))

    def ventilationHeterogeneity(self, parameterArray):
        # Keep just the points that are in the range (-1000, 0]
        arr = parameterArray[((parameterArray > -1000) & (parameterArray <= 0))]
        # Convert to float to apply the formula
        arr = arr.astype(np.float)
        # Apply formula
        arr = -arr / (arr + 1000)
        arr **= (1/3.0)
        return arr.std()

    def _moment(self, a, moment=1, axis=0):
        # Modified from SciPy module
        if moment == 1:
            return np.float64(0.0)
        else:
            mn = np.expand_dims(np.mean(a, axis), axis)
            s = np.power((a - mn), moment)
            return np.mean(s, axis)

    def skewnessValue(self, a, axis=0):
        # Modified from SciPy module
        # Computes the skewness of a dataset

        m2 = self._moment(a, 2, axis)
        m3 = self._moment(a, 3, axis)

        # Control Flow: if m2==0 then vals = 0; else vals = m3/m2**1.5
        zero = (m2 == 0)
        vals = np.where(zero, 0, m3 / m2 ** 1.5)

        if vals.ndim == 0:
            return vals.item()
        return vals

    def kurtosisValue(self, a, axis=0, fisher=True):
        # Modified from SciPy module

        m2 = self._moment(a, 2, axis)
        m4 = self._moment(a, 4, axis)
        zero = (m2 == 0)

        # Set Floating-Point Error Handling
        olderr = np.seterr(all='ignore')
        try:
            vals = np.where(zero, 0, m4 / m2 ** 2.0)
        finally:
            np.seterr(**olderr)
        if vals.ndim == 0:
            vals = vals.item()  # array scalar

        if fisher:
            return vals - 3
        else:
            return vals

    def varianceValue(self, parameterArray):
        return (np.std(parameterArray) ** 2)

    def uniformityValue(self, bins):
        return (np.sum(bins ** 2))

    def laa950PercValue(self, parameterArray):
        arr = (parameterArray <= -950)
        return 100.0*arr.sum()/np.prod(arr.shape)

    def laa910PercValue(self, parameterArray):
        arr = (parameterArray <= -910)
        return 100.0*arr.sum()/np.prod(arr.shape)

    def laa856PercValue(self, parameterArray):
        arr = (parameterArray <= -856)
        return 100.0*arr.sum()/np.prod(arr.shape)

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        # Evaluate dictionary elements corresponding to user-selected keys
        if not self.keys:
            return (self.firstOrderStatistics)

        if printTiming:
            for key in self.keys:
                t1 = time.time()
                self.firstOrderStatistics[key] = eval(self.firstOrderStatistics[key])
                self.firstOrderStatisticsTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()

            return self.firstOrderStatistics, self.firstOrderStatisticsTiming
        else:
            for key in self.keys:
                self.firstOrderStatistics[key] = eval(self.firstOrderStatistics[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.firstOrderStatistics

class GeometricalMeasures:
    def __init__(self, labelNodeSpacing, parameterMatrix, parameterMatrixCoordinates, parameterValues, allKeys):
        # need non-linear scaling of surface heights for normalization (reduce computational time)
        self.GeometricalMeasures = collections.OrderedDict()
        self.GeometricalMeasuresTiming = collections.OrderedDict()
        self.GeometricalMeasures[
            "Extruded Surface Area"] = "self.extrudedSurfaceArea(self.labelNodeSpacing, self.extrudedMatrix, self.extrudedMatrixCoordinates, self.parameterValues)"
        self.GeometricalMeasures[
            "Extruded Volume"] = "self.extrudedVolume(self.extrudedMatrix, self.extrudedMatrixCoordinates, self.cubicMMPerVoxel)"
        self.GeometricalMeasures[
            "Extruded Surface:Volume Ratio"] = "self.extrudedSurfaceVolumeRatio(self.labelNodeSpacing, self.extrudedMatrix, self.extrudedMatrixCoordinates, self.parameterValues, self.cubicMMPerVoxel)"

        self.labelNodeSpacing = labelNodeSpacing
        self.parameterMatrix = parameterMatrix
        self.parameterMatrixCoordinates = parameterMatrixCoordinates
        self.parameterValues = parameterValues
        self.keys = set(allKeys).intersection(self.GeometricalMeasures.keys())

        if self.keys:
            self.cubicMMPerVoxel = reduce(lambda x, y: x * y, labelNodeSpacing)
            self.extrudedMatrix, self.extrudedMatrixCoordinates = self.extrudeMatrix(self.parameterMatrix,
                                                                                     self.parameterMatrixCoordinates,
                                                                                     self.parameterValues)

    def extrudedSurfaceArea(self, labelNodeSpacing, extrudedMatrix, extrudedMatrixCoordinates, parameterValues):
        x, y, z = labelNodeSpacing

        # surface areas of directional connections
        xz = x * z
        yz = y * z
        xy = x * y
        fourD = (2 * xy + 2 * xz + 2 * yz)

        totalVoxelSurfaceArea4D = (2 * xy + 2 * xz + 2 * yz + 2 * fourD)
        totalSA = parameterValues.size * totalVoxelSurfaceArea4D

        # in matrixSACoordinates
        # i: height (z), j: vertical (y), k: horizontal (x), l: 4th or extrusion dimension
        i, j, k, l = 0, 0, 0, 0
        extrudedSurfaceArea = 0

        # vectorize
        for i, j, k, l_slice in zip(*extrudedMatrixCoordinates):
            for l in range(l_slice.start, l_slice.stop):
                fxy = np.array([extrudedMatrix[i + 1, j, k, l], extrudedMatrix[i - 1, j, k, l]]) == 0
                fyz = np.array([extrudedMatrix[i, j + 1, k, l], extrudedMatrix[i, j - 1, k, l]]) == 0
                fxz = np.array([extrudedMatrix[i, j, k + 1, l], extrudedMatrix[i, j, k - 1, l]]) == 0
                f4d = np.array([extrudedMatrix[i, j, k, l + 1], extrudedMatrix[i, j, k, l - 1]]) == 0

                extrudedElementSurface = (np.sum(fxz) * xz) + (np.sum(fyz) * yz) + (np.sum(fxy) * xy) + (
                np.sum(f4d) * fourD)
                extrudedSurfaceArea += extrudedElementSurface
        return (extrudedSurfaceArea)

    def extrudedVolume(self, extrudedMatrix, extrudedMatrixCoordinates, cubicMMPerVoxel):
        extrudedElementsSize = extrudedMatrix[np.where(extrudedMatrix == 1)].size
        return (extrudedElementsSize * cubicMMPerVoxel)

    def extrudedSurfaceVolumeRatio(self, labelNodeSpacing, extrudedMatrix, extrudedMatrixCoordinates, parameterValues,
                                   cubicMMPerVoxel):
        extrudedSurfaceArea = self.extrudedSurfaceArea(labelNodeSpacing, extrudedMatrix, extrudedMatrixCoordinates,
                                                       parameterValues)
        extrudedVolume = self.extrudedVolume(extrudedMatrix, extrudedMatrixCoordinates, cubicMMPerVoxel)
        return (extrudedSurfaceArea / extrudedVolume)

    def extrudeMatrix(self, parameterMatrix, parameterMatrixCoordinates, parameterValues):
        # extrude 3D image into a binary 4D array with the intensity or parameter value as the 4th Dimension
        # need to normalize CT images with a shift of 120 Hounsfield units

        parameterValues = np.abs(parameterValues)

        # maximum intensity/parameter value appended as shape of 4th dimension
        extrudedShape = parameterMatrix.shape + (np.max(parameterValues),)

        # pad shape by 1 unit in all 8 directions
        extrudedShape = tuple(map(operator.add, extrudedShape, [2, 2, 2, 2]))

        extrudedMatrix = np.zeros(extrudedShape)
        extrudedMatrixCoordinates = tuple(map(operator.add, parameterMatrixCoordinates, ([1, 1, 1]))) + (
        np.array([slice(1, value + 1) for value in parameterValues]),)
        for slice4D in zip(*extrudedMatrixCoordinates):
            extrudedMatrix[slice4D] = 1
        return (extrudedMatrix, extrudedMatrixCoordinates)

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        # Evaluate dictionary elements corresponding to user-selected keys
        if not self.keys:
            return (self.GeometricalMeasures)

        if printTiming:
            import time
            for key in self.keys:
                t1 = time.time()
                self.GeometricalMeasures[key] = eval(self.GeometricalMeasures[key])
                self.GeometricalMeasuresTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()

            return self.GeometricalMeasures, self.GeometricalMeasuresTiming
        else:
            for key in self.keys:
                self.GeometricalMeasures[key] = eval(self.GeometricalMeasures[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.GeometricalMeasures

class MorphologyStatistics:
    def __init__(self, labelNodeSpacing, matrixSA, matrixSACoordinates, matrixSAValues, allKeys):
        self.morphologyStatistics = collections.OrderedDict()
        self.morphologyStatisticsTiming = collections.OrderedDict()
        self.morphologyStatistics["Volume mm^3"] = 'self.volumeMM3(self.matrixSAValues, self.cubicMMPerVoxel)'
        self.morphologyStatistics[
            "Volume cc"] = 'self.volumeCC(self.matrixSAValues, self.cubicMMPerVoxel, self.ccPerCubicMM)'
        self.morphologyStatistics[
            "Surface Area mm^2"] = 'self.surfaceArea(self.matrixSA, self.matrixSACoordinates, self.matrixSAValues, self.labelNodeSpacing)'
        self.morphologyStatistics[
            "Surface:Volume Ratio"] = 'self.surfaceVolumeRatio(self.morphologyStatistics["Surface Area mm^2"], self.morphologyStatistics["Volume mm^3"])'
        self.morphologyStatistics[
            "Compactness 1"] = 'self.compactness1(self.morphologyStatistics["Surface Area mm^2"], self.morphologyStatistics["Volume mm^3"])'
        self.morphologyStatistics[
            "Compactness 2"] = 'self.compactness2(self.morphologyStatistics["Surface Area mm^2"], self.morphologyStatistics["Volume mm^3"])'
        self.morphologyStatistics[
            "Maximum 3D Diameter"] = 'self.maximum3DDiameter(self.labelNodeSpacing, self.matrixSA, self.matrixSACoordinates)'
        self.morphologyStatistics[
            "Spherical Disproportion"] = 'self.sphericalDisproportion(self.morphologyStatistics["Surface Area mm^2"], self.morphologyStatistics["Volume mm^3"])'
        self.morphologyStatistics[
            "Sphericity"] = 'self.sphericityValue(self.morphologyStatistics["Surface Area mm^2"], self.morphologyStatistics["Volume mm^3"])'

        self.keys = set(allKeys).intersection(self.morphologyStatistics.keys())

        self.labelNodeSpacing = labelNodeSpacing
        self.matrixSA = matrixSA
        self.matrixSACoordinates = matrixSACoordinates
        self.matrixSAValues = matrixSAValues


        self.cubicMMPerVoxel = reduce(lambda x, y: x * y, self.labelNodeSpacing)
        self.ccPerCubicMM = 0.001

    def volumeMM3(self, matrixSA, cubicMMPerVoxel):
        return (matrixSA.size * cubicMMPerVoxel)

    def volumeCC(self, matrixSA, cubicMMPerVoxel, ccPerCubicMM):
        return (matrixSA.size * cubicMMPerVoxel * ccPerCubicMM)

    def surfaceArea(self, a, matrixSACoordinates, matrixSAValues, labelNodeSpacing):
        x, y, z = labelNodeSpacing
        xz = x * z
        yz = y * z
        xy = x * y
        voxelTotalSA = (2 * xy + 2 * xz + 2 * yz)
        totalSA = matrixSAValues.size * voxelTotalSA

        # in matrixSACoordinates
        # i corresponds to height (z)
        # j corresponds to vertical (y)
        # k corresponds to horizontal (x)

        i, j, k = 0, 0, 0
        surfaceArea = 0
        for voxel in range(0, matrixSAValues.size):
            i, j, k = matrixSACoordinates[0][voxel], matrixSACoordinates[1][voxel], matrixSACoordinates[2][voxel]
            fxy = (np.array([a[i + 1, j, k], a[i - 1, j, k]]) == 0)  # evaluate to 1 if true, 0 if false
            fyz = (np.array([a[i, j + 1, k], a[i, j - 1, k]]) == 0)  # evaluate to 1 if true, 0 if false
            fxz = (np.array([a[i, j, k + 1], a[i, j, k - 1]]) == 0)  # evaluate to 1 if true, 0 if false
            surface = (np.sum(fxz) * xz) + (np.sum(fyz) * yz) + (np.sum(fxy) * xy)
            surfaceArea += surface
        return (surfaceArea)

    def surfaceVolumeRatio(self, surfaceArea, volumeMM3):
        return (surfaceArea / volumeMM3)

    def compactness1(self, surfaceArea, volumeMM3):
        return ((volumeMM3) / ((surfaceArea) ** (2 / 3.0) * math.sqrt(math.pi)))

    def compactness2(self, surfaceArea, volumeMM3):
        return ((36 * math.pi) * ((volumeMM3) ** 2) / ((surfaceArea) ** 3))

    def maximum3DDiameter(self, labelNodeSpacing, matrixSA, matrixSACoordinates):
        # largest pairwise euclidean distance between tumor surface voxels

        x, y, z = labelNodeSpacing

        minBounds = np.array(
            [np.min(matrixSACoordinates[0]), np.min(matrixSACoordinates[1]), np.min(matrixSACoordinates[2])])
        maxBounds = np.array(
            [np.max(matrixSACoordinates[0]), np.max(matrixSACoordinates[1]), np.max(matrixSACoordinates[2])])

        a = np.array(zip(*matrixSACoordinates))
        edgeVoxelsMinCoords = np.vstack(
            [a[a[:, 0] == minBounds[0]], a[a[:, 1] == minBounds[1]], a[a[:, 2] == minBounds[2]]]) * [z, y, x]
        edgeVoxelsMaxCoords = np.vstack(
            [(a[a[:, 0] == maxBounds[0]] + 1), (a[a[:, 1] == maxBounds[1]] + 1), (a[a[:, 2] == maxBounds[2]] + 1)]) * [
                                  z, y, x]

        maxDiameter = 1
        for voxel1 in edgeVoxelsMaxCoords:
            for voxel2 in edgeVoxelsMinCoords:
                voxelDistance = np.sqrt(np.sum((voxel2 - voxel1) ** 2))
                if voxelDistance > maxDiameter:
                    maxDiameter = voxelDistance
        return (maxDiameter)

    def sphericalDisproportion(self, surfaceArea, volumeMM3):
        R = ((0.75 * (volumeMM3)) / (math.pi) ** (1 / 3.0))
        return ((surfaceArea) / (4 * math.pi * (R ** 2)))

    def sphericityValue(self, surfaceArea, volumeMM3):
        return (((math.pi) ** (1 / 3.0) * (6 * volumeMM3) ** (2 / 3.0)) / (surfaceArea))

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        # Evaluate dictionary elements corresponding to user-selected keys
        if not self.keys:
            return (self.morphologyStatistics)
        if len(self.matrixSA) == 0:
            for key in self.keys:
                self.morphologyStatistics[key] = 0
        else:
            # Volume and Surface Area are pre-calculated even if only one morphology metric is user-selected
            if printTiming:
                import time
                t1 = time.time()
            self.morphologyStatistics["Volume mm^3"] = eval(self.morphologyStatistics["Volume mm^3"])
            if printTiming:
                self.morphologyStatisticsTiming["Volume mm^3"] = time.time() - t1
            self.morphologyStatistics["Surface Area mm^2"] = eval(self.morphologyStatistics["Surface Area mm^2"])
            if printTiming:
                t1 = time.time()
                self.morphologyStatisticsTiming["Surface Area mm^2"] = time.time() - t1

            if printTiming:
                for key in self.keys:
                    if isinstance(self.morphologyStatistics[key], basestring):
                        t1 = time.time()
                        self.morphologyStatistics[key] = eval(self.morphologyStatistics[key])
                        self.morphologyStatisticsTiming[key] = time.time() - t1
                        if checkStopProcessFunction is not None:
                            checkStopProcessFunction()
                return self.morphologyStatistics, self.morphologyStatisticsTiming
            else:
                for key in self.keys:
                    if isinstance(self.morphologyStatistics[key], basestring):
                        self.morphologyStatistics[key] = eval(self.morphologyStatistics[key])
                        if checkStopProcessFunction is not None:
                            checkStopProcessFunction()
                return self.morphologyStatistics

class ParenchymalVolume:
    def __init__(self, parenchymaLabelmapArray, sphereWithoutTumorLabelmapArray, spacing, keysToAnalyze=None):
        """ Parenchymal volume study.
        Compare each ones of the different labels in the original labelmap with the volume of the area of interest
        :param parenchymaLabelmapArray: original labelmap for the whole volume node
        :param sphereWithoutTumorLabelmapArray: labelmap array that contains the sphere to study without the tumor
        :param spacing: tuple of volume spacing
        :param keysToAnalyze: list of strings with the types of emphysema it's going to be analyzed. When None,
            all the types will be analyzed
        """
        self.parenchymaLabelmapArray = parenchymaLabelmapArray
        self.sphereWithoutTumorLabelmapArray = sphereWithoutTumorLabelmapArray
        self.spacing = spacing
        self.parenchymalVolumeStatistics = collections.OrderedDict()
        self.parenchymalVolumeStatisticsTiming = collections.OrderedDict()

        allKeys = self.getAllEmphysemaTypes().keys()
        if keysToAnalyze is not None:
            self.keysToAnalyze = keysToAnalyze.intersection(allKeys)
        else:
            self.keysToAnalyze = self.getAllEmphysemaTypes().keys()

    @staticmethod
    def getAllEmphysemaTypes():
        """ All emphysema types and values
        :return: dictionary of Type(string)-[numeric_code, description]
        """
        return {
            "Emphysema": 5,
            "Mild paraseptal emphysema": 10,
            "Moderate paraseptal emphysema": 11,
            "Severe paraseptal emphysema": 12,
            "Mild centrilobular emphysema": 16,
            "Moderate centrilobular emphysema": 17,
            "Severe centilobular emphysema": 18,
            "Mild panlobular emphysema": 19,
            "Moderate panlobular emphysema": 20,
            "Severe panlobular emphysema": 21
        }

    @staticmethod
    def getAllEmphysemaDescriptions():
        return ParenchymalVolume.getAllEmphysemaTypes().keys()

    def analyzeType(self, code):
        print("DEBUG: analyze code {0}.".format(code))
        # Calculate volume for the studied ROI (tumor)
        totalVolume = np.sum(self.parenchymaLabelmapArray == code)
        if totalVolume == 0:
            return 0

        # Calculate total volume in the sphere for this emphysema type
        sphereVolume = np.sum(self.parenchymaLabelmapArray[self.sphereWithoutTumorLabelmapArray] == code)

        # Result: SV / PV
        return float(sphereVolume) / totalVolume

    def EvaluateFeatures(self, printTiming = False, checkStopProcessFunction=None):
        # Evaluate dictionary elements corresponding to user-selected keys
        types = self.getAllEmphysemaTypes()

        if not printTiming:
            for key in self.keysToAnalyze:
                self.parenchymalVolumeStatistics[key] = self.analyzeType(types[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.parenchymalVolumeStatistics

        else:
            import time
            t1 = time.time()
            for key in self.keysToAnalyze:
                self.parenchymalVolumeStatistics[key] = self.analyzeType(types[key])
                self.parenchymalVolumeStatisticsTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.parenchymalVolumeStatistics, self.parenchymalVolumeStatisticsTiming


class RenyiDimensions:
    def __init__(self, matrixPadded, matrixPaddedCoordinates, allKeys):
        self.renyiDimensions = collections.OrderedDict()
        self.renyiDimensions[
            "Box-Counting Dimension"] = "self.renyiDimension(self.matrixPadded, self.matrixPaddedCoordinates, 0)"
        self.renyiDimensions[
            "Information Dimension"] = "self.renyiDimension(self.matrixPadded, self.matrixPaddedCoordinates, 1)"
        self.renyiDimensions[
            "Correlation Dimension"] = "self.renyiDimension(self.matrixPadded, self.matrixPaddedCoordinates, 2)"

        self.renyiDimensionTiming = collections.OrderedDict()

        self.matrixPadded = matrixPadded
        self.matrixPaddedCoordinates = matrixPaddedCoordinates
        self.allKeys = allKeys

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        self.checkStopProcessFunction = checkStopProcessFunction
        keys = set(self.allKeys).intersection(self.renyiDimensions.keys())
        if not printTiming:
            if not keys:
                return self.renyiDimensions

            # Evaluate dictionary elements corresponding to user selected keys
            for key in keys:
                self.renyiDimensions[key] = eval(self.renyiDimensions[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.renyiDimensions
        else:
            if not keys:
                return self.renyiDimensions, self.renyiDimensionTiming

            import time
            # Evaluate dictionary elements corresponding to user selected keys
            for key in keys:
                t1 = time.time()
                self.renyiDimensions[key] = eval(self.renyiDimensions[key])
                self.renyiDimensionTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.renyiDimensions, self.renyiDimensionTiming

    def renyiDimension(self, c, matrixCoordinatesPadded, q=0):
        # computes renyi dimensions for q = 0,1,2 (box-count(default, q=0), information(q=1), and correlation dimensions(q=2))
        # for a padded 3D input array or matrix, c, and the coordinates of values in c, matrixCoordinatesPadded.
        # c must be padded to a cube with shape equal to next greatest power of two
        # i.e. a 3D array with shape: (3,13,9) is padded to shape: (16,16,16)

        # exception for np.sum(c) = 0?
        c = c / float(np.sum(c))
        maxDim = c.shape[0]
        p = int(np.log2(maxDim))
        n = np.zeros(p + 1)
        eps = np.spacing(1)

        # Initialize N(s) value at the finest/voxel-level scale
        if (q == 1):
            n[p] = np.sum(c[matrixCoordinatesPadded] * np.log(1 / (c[matrixCoordinatesPadded] + eps)))
            if np.isnan(n[p]):
                n[p]=0.0
        else:
            n[p] = np.sum(c[matrixCoordinatesPadded] ** q)

        for g in range(p - 1, -1, -1):
            siz = 2 ** (p - g)
            siz2 = int(round(siz / 2))
            for i in range(0, maxDim - siz + 1, siz):
                for j in range(0, maxDim - siz + 1, siz):
                    for k in range(0, maxDim - siz + 1, siz):
                        box = np.array([c[i, j, k], c[i + siz2, j, k], c[i, j + siz2, k], c[i + siz2, j + siz2, k],
                                           c[i, j, k + siz2], c[i + siz2, j, k + siz2], c[i, j + siz2, k + siz2],
                                           c[i + siz2, j + siz2, k + siz2]])
                        c[i, j, k] = np.any(box != 0) if (q == 0) else np.sum(box) ** q
                    if self.checkStopProcessFunction is not None:
                        self.checkStopProcessFunction()
                        # print (i, j, k, '                ', c[i,j,k])
            pi = c[0:(maxDim - siz + 1):siz, 0:(maxDim - siz + 1):siz, 0:(maxDim - siz + 1):siz]
            
            #For some reason pi has nan. Let set those places to zero. I do not know if this is the right choice.
            pi[np.isnan(pi)]=0
            if (q == 1):
                n[g] = np.sum(pi * np.log(1 / (pi + eps)))
                if np.isnan(n[g]):
                    n[g]=0.0
            else:
                n[g] = np.sum(pi)
            #print ('p, g, siz, siz2', p, g, siz, siz2, '         n[g]: ', n[g])

        r = np.log(2.0 ** (np.arange(p + 1)))  # log(1/scale)
        scaleMatrix = np.array([r, np.ones(p + 1)])
        # print ('n(s): ', n)
        # print ('log (1/s): ', r)

        if (q != 1):
            n = (1 / float(1 - q)) * np.log(n)
        renyiDimension = np.linalg.lstsq(scaleMatrix.T, n)[0][0]

        return (renyiDimension)

class TextureGLCM:
    def __init__(self, grayLevels, numGrayLevels, parameterMatrix, parameterMatrixCoordinates, parameterValues,
                 allKeys, checkStopProcessFunction):
        self.textureFeaturesGLCM = collections.OrderedDict()
        self.textureFeaturesGLCMTiming = collections.OrderedDict()

        self.textureFeaturesGLCM["Autocorrelation"] = "self.autocorrelationGLCM(self.P_glcm, self.prodMatrix)"
        self.textureFeaturesGLCM[
            "Cluster Prominence"] = "self.clusterProminenceGLCM(self.P_glcm, self.sumMatrix, self.ux, self.uy)"
        self.textureFeaturesGLCM[
            "Cluster Shade"] = "self.clusterShadeGLCM(self.P_glcm, self.sumMatrix, self.ux, self.uy)"
        self.textureFeaturesGLCM[
            "Cluster Tendency"] = "self.clusterTendencyGLCM(self.P_glcm, self.sumMatrix, self.ux, self.uy)"
        self.textureFeaturesGLCM["Contrast"] = "self.contrastGLCM(self.P_glcm, self.diffMatrix)"
        self.textureFeaturesGLCM[
            "Correlation"] = "self.correlationGLCM(self.P_glcm, self.prodMatrix, self.ux, self.uy, self.sigx, self.sigy)"
        self.textureFeaturesGLCM["Difference Entropy"] = "self.differenceEntropyGLCM(self.pxSuby, self.eps)"
        self.textureFeaturesGLCM["Dissimilarity"] = "self.dissimilarityGLCM(self.P_glcm, self.diffMatrix)"
        self.textureFeaturesGLCM["Energy (GLCM)"] = "self.energyGLCM(self.P_glcm)"
        self.textureFeaturesGLCM["Entropy(GLCM)"] = "self.entropyGLCM(self.P_glcm, self.pxy, self.eps)"
        self.textureFeaturesGLCM["Homogeneity 1"] = "self.homogeneity1GLCM(self.P_glcm, self.diffMatrix)"
        self.textureFeaturesGLCM["Homogeneity 2"] = "self.homogeneity2GLCM(self.P_glcm, self.diffMatrix)"
        self.textureFeaturesGLCM["IMC1"] = "self.imc1GLCM(self.HXY, self.HXY1, self.HX, self.HY)"
        # self.textureFeaturesGLCM["IMC2"] = "sum(imc2)/len(imc2)" #"self.imc2GLCM(self,)"  # produces a calculation error
        self.textureFeaturesGLCM["IDMN"] = "self.idmnGLCM(self.P_glcm, self.diffMatrix, self.Ng)"
        self.textureFeaturesGLCM["IDN"] = "self.idnGLCM(self.P_glcm, self.diffMatrix, self.Ng)"
        self.textureFeaturesGLCM["Inverse Variance"] = "self.inverseVarianceGLCM(self.P_glcm, self.diffMatrix, self.Ng)"
        self.textureFeaturesGLCM["Maximum Probability"] = "self.maximumProbabilityGLCM(self.P_glcm)"
        self.textureFeaturesGLCM["Sum Average"] = "self.sumAverageGLCM(self.pxAddy, self.kValuesSum)"
        self.textureFeaturesGLCM["Sum Entropy"] = "self.sumEntropyGLCM(self.pxAddy, self.eps)"
        self.textureFeaturesGLCM["Sum Variance"] = "self.sumVarianceGLCM(self.pxAddy, self.kValuesSum)"
        self.textureFeaturesGLCM["Variance (GLCM)"] = "self.varianceGLCM(self.P_glcm, self.ivector, self.u)"

        self.grayLevels = grayLevels
        self.parameterMatrix = parameterMatrix
        self.parameterMatrixCoordinates = parameterMatrixCoordinates
        self.parameterValues = parameterValues
        self.Ng = numGrayLevels
        self.keys = set(allKeys).intersection(self.textureFeaturesGLCM.keys())
        # Callback function to stop the process if the user decided so. CalculateCoefficients can take a long time to run...
        self.checkStopProcessFunction = checkStopProcessFunction

    def CalculateCoefficients(self, printTiming=False):
        """ Calculate generic coefficients that will be reused in different markers
        IMPORTANT!! This method is VERY inefficient when the nodule is big (because
        of the function calculate_glcm at least). If these
        statistics are required it would probably need some optimizations
        :return:
        """
        # generate container for GLCM Matrices, self.P_glcm
        # make distance an optional parameter, as in: distances = np.arange(parameter)
        distances = np.array([1])
        directions = 26
        self.P_glcm = np.zeros((self.Ng, self.Ng, distances.size, directions))
        t1 = time.time()
        self.P_glcm = self.calculate_glcm(self.grayLevels, self.parameterMatrix, self.parameterMatrixCoordinates,
                                          distances, directions, self.Ng, self.P_glcm)
        if printTiming:
            print("- Time to calculate glmc matrix: {0} secs".format(time.time() - t1))
        # make each GLCM symmetric an optional parameter
        # if symmetric:
        # Pt = np.transpose(P, (1, 0, 2, 3))
        # P = P + Pt

        ##Calculate GLCM Coefficients
        self.ivector = np.arange(1, self.Ng + 1)  # shape = (self.Ng, distances.size, directions)
        self.jvector = np.arange(1, self.Ng + 1)  # shape = (self.Ng, distances.size, directions)
        self.eps = np.spacing(1)

        self.prodMatrix = np.multiply.outer(self.ivector, self.jvector)  # shape = (self.Ng, self.Ng)
        self.sumMatrix = np.add.outer(self.ivector, self.jvector)  # shape = (self.Ng, self.Ng)
        self.diffMatrix = np.absolute(np.subtract.outer(self.ivector, self.jvector))  # shape = (self.Ng, self.Ng)
        self.kValuesSum = np.arange(2, (self.Ng * 2) + 1)  # shape = (2*self.Ng-1)
        self.kValuesDiff = np.arange(0, self.Ng)  # shape = (self.Ng-1)

        # shape = (distances.size, directions)
        self.u = self.P_glcm.mean(0).mean(0)
        # marginal row probabilities #shape = (self.Ng, distances.size, directions)
        self.px = self.P_glcm.sum(1)
        # marginal column probabilities #shape = (self.Ng, distances.size, directions)
        self.py = self.P_glcm.sum(0)

        # shape = (distances.size, directions)
        self.ux = self.px.mean(0)
        # shape = (distances.size, directions)
        self.uy = self.py.mean(0)

        # shape = (distances.size, directions)
        self.sigx = self.px.std(0)
        # shape = (distances.size, directions)
        self.sigy = self.py.std(0)

        # shape = (2*self.Ng-1, distances.size, directions)
        self.pxAddy = np.array([np.sum(self.P_glcm[self.sumMatrix == k], 0) for k in self.kValuesSum])
        # shape = (self.Ng, distances.size, directions)
        self.pxSuby = np.array([np.sum(self.P_glcm[self.diffMatrix == k], 0) for k in self.kValuesDiff])

        # entropy of self.px #shape = (distances.size, directions)
        self.HX = (-1) * np.sum((self.px * np.where(self.px != 0, np.log2(self.px), np.log2(self.eps))), 0)
        # entropy of py #shape = (distances.size, directions)
        self.HY = (-1) * np.sum((self.py * np.where(self.py != 0, np.log2(self.py), np.log2(self.eps))), 0)
        # shape = (distances.size, directions)
        self.HXY = (-1) * np.sum(
            np.sum((self.P_glcm * np.where(self.P_glcm != 0, np.log2(self.P_glcm), np.log2(self.eps))), 0),
            0)

        self.pxy = np.zeros(self.P_glcm.shape)  # shape = (self.Ng, self.Ng, distances.size, directions)
        for a in range(directions):
            for g in range(distances.size):
                self.pxy[:, :, g, a] = np.multiply.outer(self.px[:, g, a], self.py[:, g, a])

        self.HXY1 = (-1) * np.sum(
            np.sum((self.P_glcm * np.where(self.pxy != 0, np.log2(self.pxy), np.log2(self.eps))), 0),
            0)  # shape = (distances.size, directions)
        self.HXY2 = (-1) * np.sum(
            np.sum((self.pxy * np.where(self.pxy != 0, np.log2(self.pxy), np.log2(self.eps))), 0),
            0)  # shape = (distances.size, directions)
        if printTiming:
            print("- Time to calculate total glmc coefficients: {0} secs".format(time.time() - t1))

    def autocorrelationGLCM(self, P_glcm, prodMatrix, meanFlag=True):
        ac = np.sum(np.sum(P_glcm * prodMatrix[:, :, None, None], 0), 0)
        if meanFlag:
            return (ac.mean())
        else:
            return ac

    def clusterProminenceGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        cp = np.sum(
            np.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 4)), 0),
            0)
        if meanFlag:
            return (cp.mean())
        else:
            return cp

    def clusterShadeGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        cs = np.sum(
            np.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 3)), 0),
            0)
        if meanFlag:
            return (cs.mean())
        else:
            return cs

    def clusterTendencyGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        ct = np.sum(
            np.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 2)), 0),
            0)
        if meanFlag:
            return (ct.mean())
        else:
            return ct

    def contrastGLCM(self, P_glcm, diffMatrix, meanFlag=True):
        cont = np.sum(np.sum((P_glcm * (diffMatrix[:, :, None, None] ** 2)), 0), 0)
        if meanFlag:
            return (cont.mean())
        else:
            return cont

    def correlationGLCM(self, P_glcm, prodMatrix, ux, uy, sigx, sigy, meanFlag=True):
        # Need to validate function
        uxy = ux * uy
        sigxy = sigx * sigy
        corr = np.sum(
            np.sum(((P_glcm * prodMatrix[:, :, None, None] - uxy[None, None, :, :]) / (sigxy[None, None, :, :])), 0),
            0)
        if meanFlag:
            return (corr.mean())
        else:
            return corr

    def differenceEntropyGLCM(self, pxSuby, eps, meanFlag=True):
        difent = np.sum((pxSuby * np.where(pxSuby != 0, np.log2(pxSuby), np.log2(eps))), 0)
        if meanFlag:
            return (difent.mean())
        else:
            return difent

    def dissimilarityGLCM(self, P_glcm, diffMatrix, meanFlag=True):
        dis = np.sum(np.sum((P_glcm * diffMatrix[:, :, None, None]), 0), 0)
        if meanFlag:
            return (dis.mean())
        else:
            return dis

    def energyGLCM(self, P_glcm, meanFlag=True):
        ene = np.sum(np.sum((P_glcm ** 2), 0), 0)
        if meanFlag:
            return (ene.mean())
        else:
            return ene

    def entropyGLCM(self, P_glcm, pxy, eps, meanFlag=True):
        ent = -1 * np.sum(np.sum((P_glcm * np.where(pxy != 0, np.log2(pxy), np.log2(eps))), 0), 0)
        if meanFlag:
            return (ent.mean())
        else:
            return ent

    def homogeneity1GLCM(self, P_glcm, diffMatrix, meanFlag=True):
        homo1 = np.sum(np.sum((P_glcm / (1 + diffMatrix[:, :, None, None])), 0), 0)
        if meanFlag:
            return (homo1.mean())
        else:
            return homo1

    def homogeneity2GLCM(self, P_glcm, diffMatrix, meanFlag=True):
        homo2 = np.sum(np.sum((P_glcm / (1 + diffMatrix[:, :, None, None] ** 2)), 0), 0)
        if meanFlag:
            return (homo2.mean())
        else:
            return homo2

    def imc1GLCM(self, HXY, HXY1, HX, HY, meanFlag=True):
        imc1 = (self.HXY - self.HXY1) / np.max(([self.HX, self.HY]), 0)
        if meanFlag:
            return (imc1.mean())
        else:
            return imc1

            # def imc2GLCM(self,):
            # imc2[g,a] = ( 1-np.e**(-2*(HXY2[g,a]-HXY[g,a])) )**(0.5) #nan value too high

            # produces Nan(square root of a negative)
            # exponent = decimal.Decimal( -2*(HXY2[g,a]-self.HXY[g,a]) )
            # imc2.append( ( decimal.Decimal(1)-decimal.Decimal(np.e)**(exponent) )**(decimal.Decimal(0.5)) )

            # if meanFlag:
            # return (homo2.mean())
            # else:
            # return homo2

    def idmnGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        idmn = np.sum(np.sum((P_glcm / (1 + ((diffMatrix[:, :, None, None] ** 2) / (Ng ** 2)))), 0), 0)
        if meanFlag:
            return (idmn.mean())
        else:
            return idmn

    def idnGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        idn = np.sum(np.sum((P_glcm / (1 + (diffMatrix[:, :, None, None] / Ng))), 0), 0)
        if meanFlag:
            return (idn.mean())
        else:
            return idn

    def inverseVarianceGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        maskDiags = np.ones(diffMatrix.shape, dtype=bool)
        maskDiags[np.diag_indices(Ng)] = False
        inv = np.sum((P_glcm[maskDiags] / (diffMatrix[:, :, None, None] ** 2)[maskDiags]), 0)
        if meanFlag:
            return (inv.mean())
        else:
            return inv

    def maximumProbabilityGLCM(self, P_glcm, meanFlag=True):
        maxprob = P_glcm.max(0).max(0)
        if meanFlag:
            return (maxprob.mean())
        else:
            return maxprob

    def sumAverageGLCM(self, pxAddy, kValuesSum, meanFlag=True):
        sumavg = np.sum((kValuesSum[:, None, None] * pxAddy), 0)
        if meanFlag:
            return (sumavg.mean())
        else:
            return sumavg

    def sumEntropyGLCM(self, pxAddy, eps, meanFlag=True):
        sumentr = (-1) * np.sum((pxAddy * np.where(pxAddy != 0, np.log2(pxAddy), np.log2(eps))), 0)
        if meanFlag:
            return (sumentr.mean())
        else:
            return sumentr

    def sumVarianceGLCM(self, pxAddy, kValuesSum, meanFlag=True):
        sumvar = np.sum((pxAddy * ((kValuesSum[:, None, None] - kValuesSum[:, None, None] * pxAddy) ** 2)), 0)
        if meanFlag:
            return (sumvar.mean())
        else:
            return sumvar

    def varianceGLCM(self, P_glcm, ivector, u, meanFlag=True):
        vari = np.sum(np.sum((P_glcm * ((ivector[:, None] - u) ** 2)[:, None, None, :]), 0), 0)
        if meanFlag:
            return (vari.mean())
        else:
            return vari

    def calculate_glcm(self, grayLevels, matrix, matrixCoordinates, distances, directions, numGrayLevels, out):
        # VERY INEFFICIENT!!
        # 26 GLCM matrices for each image for every direction from the voxel
        # (26 for each neighboring voxel from a reference voxel centered in a 3x3 cube)
        # for GLCM matrices P(i,j;gamma, a), gamma = 1, a = 1...13

        angles_idx = 0
        distances_idx = 0
        r = 0
        c = 0
        h = 0
        rows = matrix.shape[2]
        cols = matrix.shape[1]
        height = matrix.shape[0]
        row = 0
        col = 0
        height = 0

        angles = np.array([(1, 0, 0),
                              (-1, 0, 0),
                              (0, 1, 0),
                              (0, -1, 0),
                              (0, 0, 1),
                              (0, 0, -1),
                              (1, 1, 0),
                              (-1, 1, 0),
                              (1, -1, 0),
                              (-1, -1, 0),
                              (1, 0, 1),
                              (-1, 0, 1),
                              (1, 0, -1),
                              (-1, 0, -1),
                              (0, 1, 1),
                              (0, -1, 1),
                              (0, 1, -1),
                              (0, -1, -1),
                              (1, 1, 1),
                              (-1, 1, 1),
                              (1, -1, 1),
                              (1, 1, -1),
                              (-1, -1, 1),
                              (-1, 1, -1),
                              (1, -1, -1),
                              (-1, -1, -1)])

        indices = zip(*matrixCoordinates)


        for iteration in range(len(indices)):
        #for h, c, r in indices:
            h, c, r = indices[iteration]
            for angles_idx in range(directions):
                angle = angles[angles_idx]

                for distances_idx in range(distances.size):
                    distance = distances[distances_idx]

                    i = matrix[h, c, r]
                    i_idx = np.nonzero(grayLevels == i)

                    row = r + angle[2]
                    col = c + angle[1]
                    height = h + angle[0]

                    # Can introduce Parameter Option for reference voxel(i) and neighbor voxel(j):
                    # Intratumor only: i and j both must be in tumor ROI
                    # Tumor+Surrounding: i must be in tumor ROI but J does not have to be
                    if row >= 0 and row < rows and col >= 0 and col < cols:
                        if tuple((height, col, row)) in indices:
                            j = matrix[height, col, row]
                            j_idx = np.nonzero(grayLevels == j)
                            # if i >= grayLevels.min and i <= grayLevels.max and j >= grayLevels.min and j <= grayLevels.max:
                            out[i_idx, j_idx, distances_idx, angles_idx] += 1
            # Check if the user has cancelled the process
            if iteration % 10 == 0 and self.checkStopProcessFunction is not None:
                self.checkStopProcessFunction()

        return (out)

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        if not self.keys:
            if not printTiming:
                return self.textureFeaturesGLCM
            else:
                self.textureFeaturesGLCMTiming.update
                return self.textureFeaturesGLCM, self.textureFeaturesGLCMTiming
        # normalization step:
        self.CalculateCoefficients(printTiming)

        if not printTiming:
            # Evaluate dictionary elements corresponding to user selected keys
            for key in self.keys:
                self.textureFeaturesGLCM[key] = eval(self.textureFeaturesGLCM[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.textureFeaturesGLCM
        else:
            # Evaluate dictionary elements corresponding to user selected keys
            for key in self.keys:
                t1 = time.time()
                self.textureFeaturesGLCM[key] = eval(self.textureFeaturesGLCM[key])
                self.textureFeaturesGLCMTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.textureFeaturesGLCM, self.textureFeaturesGLCMTiming


class TextureGLRL:
    def __init__(self, grayLevels, numGrayLevels, parameterMatrix, parameterMatrixCoordinates, parameterValues,
                 allKeys):
        self.textureFeaturesGLRL = collections.OrderedDict()
        self.textureFeaturesGLRLTiming = collections.OrderedDict()
        self.textureFeaturesGLRL["SRE"] = "self.shortRunEmphasis(self.P_glrl, self.jvector, self.sumP_glrl)"
        self.textureFeaturesGLRL["LRE"] = "self.longRunEmphasis(self.P_glrl, self.jvector, self.sumP_glrl)"
        self.textureFeaturesGLRL["GLN"] = "self.grayLevelNonUniformity(self.P_glrl, self.sumP_glrl)"
        self.textureFeaturesGLRL["RLN"] = "self.runLengthNonUniformity(self.P_glrl, self.sumP_glrl)"
        self.textureFeaturesGLRL["RP"] = "self.runPercentage(self.P_glrl, self.Np)"
        self.textureFeaturesGLRL["LGLRE"] = "self.lowGrayLevelRunEmphasis(self.P_glrl, self.ivector, self.sumP_glrl)"
        self.textureFeaturesGLRL["HGLRE"] = "self.highGrayLevelRunEmphasis(self.P_glrl, self.ivector, self.sumP_glrl)"
        self.textureFeaturesGLRL[
            "SRLGLE"] = "self.shortRunLowGrayLevelEmphasis(self.P_glrl, self.ivector, self.jvector, self.sumP_glrl)"
        self.textureFeaturesGLRL[
            "SRHGLE"] = "self.shortRunHighGrayLevelEmphasis(self.P_glrl, self.ivector, self.jvector, self.sumP_glrl)"
        self.textureFeaturesGLRL[
            "LRLGLE"] = "self.longRunLowGrayLevelEmphasis(self.P_glrl, self.ivector, self.jvector, self.sumP_glrl)"
        self.textureFeaturesGLRL[
            "LRHGLE"] = "self.longRunHighGrayLevelEmphasis(self.P_glrl, self.ivector, self.jvector, self.sumP_glrl)"

        self.grayLevels = grayLevels
        self.parameterMatrix = parameterMatrix
        self.parameterMatrixCoordinates = parameterMatrixCoordinates
        self.parameterValues = parameterValues
        self.numGrayLevels = numGrayLevels
        self.keys = set(allKeys).intersection(self.textureFeaturesGLRL.keys())

    def CalculateCoefficients(self):
        self.angles = 13
        self.Ng = self.numGrayLevels
        self.Nr = np.max(self.parameterMatrix.shape)
        self.Np = self.parameterValues.size
        self.eps = np.spacing(1)

        self.P_glrl = np.zeros(
            (self.Ng, self.Nr, self.angles))  # maximum run length in P matrix initialized to highest gray level
        self.P_glrl = self.calculate_glrl(self.grayLevels, self.Ng, self.parameterMatrix,
                                          self.parameterMatrixCoordinates, self.angles, self.P_glrl)

        self.sumP_glrl = np.sum(np.sum(self.P_glrl, 0), 0) + self.eps
        self.ivector = np.arange(self.Ng) + 1
        self.jvector = np.arange(self.Nr) + 1

    def shortRunEmphasis(self, P_glrl, jvector, sumP_glrl, meanFlag=True):
        try:
            sre = np.sum(np.sum((P_glrl / ((jvector ** 2)[None, :, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            sre = 0
        if meanFlag:
            return (sre.mean())
        else:
            return sre

    def longRunEmphasis(self, P_glrl, jvector, sumP_glrl, meanFlag=True):
        try:
            lre = np.sum(np.sum((P_glrl * ((jvector ** 2)[None, :, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lre = 0
        if meanFlag:
            return (lre.mean())
        else:
            return lre

    def grayLevelNonUniformity(self, P_glrl, sumP_glrl, meanFlag=True):
        try:
            gln = np.sum((np.sum(P_glrl, 1) ** 2), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            gln = 0
        if meanFlag:
            return (gln.mean())
        else:
            return gln

    def runLengthNonUniformity(self, P_glrl, sumP_glrl, meanFlag=True):
        try:
            rln = np.sum((np.sum(P_glrl, 0) ** 2), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            rln = 0
        if meanFlag:
            return (rln.mean())
        else:
            return rln

    def runPercentage(self, P_glrl, Np, meanFlag=True):
        try:
            rp = np.sum(np.sum((P_glrl / (Np)), 0), 0)
        except ZeroDivisionError:
            rp = 0
        if meanFlag:
            return (rp.mean())
        else:
            return rp

    def lowGrayLevelRunEmphasis(self, P_glrl, ivector, sumP_glrl, meanFlag=True):
        try:
            lglre = np.sum(np.sum((P_glrl / ((ivector ** 2)[:, None, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lglre = 0
        if meanFlag:
            return (lglre.mean())
        else:
            return lglre

    def highGrayLevelRunEmphasis(self, P_glrl, ivector, sumP_glrl, meanFlag=True):
        try:
            hglre = np.sum(np.sum((P_glrl * ((ivector ** 2)[:, None, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            hglre = 0
        if meanFlag:
            return (hglre.mean())
        else:
            return hglre

    def shortRunLowGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            srlgle = np.sum(np.sum((P_glrl / ((jvector ** 2)[None, :, None] * (ivector ** 2)[:, None, None])), 0),
                               0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            srlgle = 0
        if meanFlag:
            return (srlgle.mean())
        else:
            return srlgle

    def shortRunHighGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            srhgle = np.sum(
                np.sum(((P_glrl * (ivector ** 2)[:, None, None]) / ((jvector ** 2)[None, :, None])), 0), 0) / (
                         sumP_glrl[None, None, :])
        except ZeroDivisionError:
            srhgle = 0
        if meanFlag:
            return (srhgle.mean())
        else:
            return srhgle

    def longRunLowGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            lrlgle = np.sum(
                np.sum(((P_glrl * (jvector ** 2)[None, :, None]) / ((ivector ** 2)[:, None, None])), 0), 0) / (
                         sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lrlgle = 0
        if meanFlag:
            return (lrlgle.mean())
        else:
            return lrlgle

    def longRunHighGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            lrhgle = np.sum(np.sum((P_glrl * (ivector ** 2)[:, None, None] * (jvector ** 2)[None, :, None]), 0),
                               0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lrhgle = 0
        if meanFlag:
            return (lrhgle.mean())
        else:
            return lrhgle

    def calculate_glrl(self, grayLevels, numGrayLevels, matrix, matrixCoordinates, angles, P_out):
        padVal = 0  # use eps or NaN to pad matrix
        matrixDiagonals = list()

        # TODO: try using itertools list merging with lists of GLRL diagonal
        # i.e.: self.heterogeneityFeatureWidgets = list(itertools.chain.from_iterable(self.featureWidgets.values()))

        # For a single direction or diagonal (aDiags, bDiags...lDiags, mDiags):
        # Generate a 1D array for each valid offset of the diagonal, a, in the range specified by lowBound and highBound
        # Convert each 1D array to a python list ( matrix.diagonal(a,,).tolist() )
        # Join lists using reduce(lamda x,y: x+y, ...) to represent all 1D arrays for the direction/diagonal
        # Use filter(lambda x: np.nonzero(x)[0].size>1, ....) to filter 1D arrays of size < 2 or value == 0 or padValue

        # Should change from nonzero() to filter for the padValue specifically (NaN, eps, etc)

        # (1,0,0), #(-1,0,0),
        aDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (1, 2, 0))])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, aDiags))

        # (0,1,0), #(0,-1,0),
        bDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (0, 2, 1))])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, bDiags))

        # (0,0,1), #(0,0,-1),
        cDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (0, 1, 2))])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, cDiags))

        # (1,1,0),#(-1,-1,0),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        dDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 1).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, dDiags))

        # (1,0,1), #(-1,0-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[2]

        eDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 2).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, eDiags))

        # (0,1,1), #(0,-1,-1),
        lowBound = -matrix.shape[1] + 1
        highBound = matrix.shape[2]

        fDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 1, 2).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, fDiags))

        # (1,-1,0), #(-1,1,0),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        gDiags = reduce(lambda x, y: x + y,
                        [matrix[:, ::-1, :].diagonal(a, 0, 1).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, gDiags))

        # (-1,0,1), #(1,0,-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[2]

        hDiags = reduce(lambda x, y: x + y,
                        [matrix[:, :, ::-1].diagonal(a, 0, 2).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, hDiags))

        # (0,1,-1), #(0,-1,1),
        lowBound = -matrix.shape[1] + 1
        highBound = matrix.shape[2]

        iDiags = reduce(lambda x, y: x + y,
                        [matrix[:, :, ::-1].diagonal(a, 1, 2).tolist() for a in range(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, iDiags))

        # (1,1,1), #(-1,-1,-1)
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        jDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix.diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
                  range(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, jDiags))

        # (-1,1,-1), #(1,-1,1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        kDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, ::-1, :].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
                  range(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, kDiags))

        # (1,1,-1), #(-1,-1,1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        lDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, :, ::-1].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
                  range(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, lDiags))

        # (-1,1,1), #(1,-1,-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        mDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, ::-1, ::-1].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
                  range(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, mDiags))

        # [n for n in mDiags if np.nonzero(n)[0].size>1] instead of filter(lambda x: np.nonzero(x)[0].size>1, mDiags)?

        # Run-Length Encoding (rle) for the 13 list of diagonals (1 list per 3D direction/angle)
        for angle in range(0, len(matrixDiagonals)):
            P = P_out[:, :, angle]
            for diagonal in matrixDiagonals[angle]:
                diagonal = np.array(diagonal, dtype='int')
                pos, = np.where(
                    np.diff(diagonal) != 0)  # can use instead of using map operator._ on np.where tuples
                pos = np.concatenate(([0], pos + 1, [len(diagonal)]))

                # a or pos[:-1] = run start #b or pos[1:] = run stop #diagonal[a] is matrix value
                # adjust condition for pos[:-1] != padVal = 0 to != padVal = eps or NaN or whatever pad value
                rle = zip([n for n in diagonal[pos[:-1]] if n != padVal], pos[1:] - pos[:-1])
                rle = [[np.where(grayLevels == x)[0][0], y - 1] for x, y in
                       rle]  # rle = map(lambda (x,y): [voxelToIndexDict[x],y-1], rle)

                # Increment GLRL matrix counter at coordinates defined by the run-length encoding
                P[zip(*rle)] += 1

        return (P_out)

    def EvaluateFeatures(self, printTiming=False, checkStopProcessFunction=None):
        if not self.keys:
            if not printTiming:
                return self.textureFeaturesGLRL
            else:
                return self.textureFeaturesGLRL, self.textureFeaturesGLRLTiming

        if not printTiming:
            self.CalculateCoefficients()
        else:
            import time
            t1 = time.time()
            self.CalculateCoefficients()
            print("- Time to calculate coefficients in GLRL: {0} seconds".format(time.time() - t1))

        if not printTiming:
            # Evaluate dictionary elements corresponding to user selected keys
            for key in self.keys:
                self.textureFeaturesGLRL[key] = eval(self.textureFeaturesGLRL[key])
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.textureFeaturesGLRL
        else:
            # Evaluate dictionary elements corresponding to user selected keys
            for key in self.keys:
                t1 = time.time()
                self.textureFeaturesGLRL[key] = eval(self.textureFeaturesGLRL[key])
                self.textureFeaturesGLRLTiming[key] = time.time() - t1
                if checkStopProcessFunction is not None:
                    checkStopProcessFunction()
            return self.textureFeaturesGLRL, self.textureFeaturesGLRLTiming


class NodulePhenotypes:

    def __init__(self, input_ct, nodule_lm, nodule_id, seed_point, l_type, segm_thresh, c_sphere_rads,b_sphere_rads,
                 feature_classes, csv_file_path, all_features,subtypes_lm):
        self.WORKING_MODE_HUMAN = 0
        self.WORKING_MODE_SMALL_ANIMAL = 1
        self.MAX_TUMOR_RADIUS = 30

        self._input_ct = input_ct
        self._nodule_lm = nodule_lm
        self._nodule_id = nodule_id
        self._seed_point = seed_point
        self._lesion_type = l_type
        self._segmentation_threshold = segm_thresh
        self._centroid_sphere_radius = c_sphere_rads
        self._boundary_sphere_radius = b_sphere_rads
        self._all_features = all_features
        self._feature_classes = feature_classes
        self._current_distance_maps = {}
        self._analysis_results = dict()
        self._csv_file_path = csv_file_path
        self._centroid_analyzedSpheres = list()
        self._boundary_analyzedSpheres = list()
        self._subtypes_lm = subtypes_lm

    def execute(self, cid, whole_lm=None):
        """ Compute all the features that are currently selected, for the nodule and/or for
        the surrounding spheres
        """
        # build list of features and feature classes based on what is checked by the user
        selectedMainFeaturesKeys = set()
        selectedFeatureKeys = set()

        try:
            # Analysis for the volume and the nodule:
            keyName = cid
            feature_widgets = collections.OrderedDict()
            for key in self._feature_classes.keys():
                feature_widgets[key] = list()

            for fc in self._feature_classes:
                for f in self._feature_classes[fc]:
                    selectedMainFeaturesKeys.add(fc)
                    selectedFeatureKeys.add(str(f))

            print("******** Nodule analysis results...")
            nodule_lm_array = sitk.GetArrayFromImage(self._nodule_lm)
            reference="Interior"
            self._analysis_results[keyName,reference] = collections.OrderedDict()
            self._analysis_results[keyName,reference] = self.runAnalysis(self._subtypes_lm, nodule_lm_array, self._analysis_results[keyName,reference],
                                                               selectedMainFeaturesKeys, selectedFeatureKeys)

            # Print analysis results
            print(self._analysis_results[keyName,reference])

            if self._centroid_sphere_radius is not None:
                print ("Radius in reference to centroid")
                self._current_distance_maps[cid] = self.getCurrentDistanceMapFromCentroid(whole_lm)
                reference="Centroid"
                for sph_rad in self._centroid_sphere_radius:
                    if sph_rad > 0.0:
                        print ("Running analysis for "+str(sph_rad))
                        self.runAnalysisSphere(cid, reference, self._current_distance_maps[cid], sph_rad,
                                       selectedMainFeaturesKeys, selectedFeatureKeys, self._subtypes_lm)
                        self._centroid_analyzedSpheres.append(sph_rad)
            
            if self._boundary_sphere_radius is not None:
                print ("Radius in reference to nodule boundary")
                self._current_distance_maps[cid] = self.getCurrentDistanceMapFromNodule(whole_lm)
                reference="Boundary"
                for sph_rad in self._boundary_sphere_radius:
                    if sph_rad > 0.0:
                        print ("Running analysis for "+str(sph_rad))
                        self.runAnalysisSphere(cid, reference, self._current_distance_maps[cid], sph_rad,
                                               selectedMainFeaturesKeys, selectedFeatureKeys, self._subtypes_lm)
                        self._boundary_analyzedSpheres.append(sph_rad)
                        


        finally:
            self.saveReport(cid)

    def runAnalysis(self, subtype_lm, n_lm_array, results_storage, feature_categories_keys, feature_keys):
        t1 = time.time()
        i_ct_array = sitk.GetArrayFromImage(self._input_ct)
        targetVoxels, targetVoxelsCoordinates = self.tumorVoxelsAndCoordinates(n_lm_array, i_ct_array)
        print("Time to calculate tumorVoxelsAndCoordinates: {0} seconds".format(time.time() - t1))
        print (np.shape(targetVoxels))
        print (np.shape(targetVoxelsCoordinates))
        # create a padded, rectangular matrix with shape equal to the shape of the tumor
        t1 = time.time()
        matrix, matrixCoordinates = self.paddedTumorMatrixAndCoordinates(targetVoxels, targetVoxelsCoordinates)
        print("Time to calculate paddedTumorMatrixAndCoordinates: {0} seconds".format(time.time() - t1))

        # get Histogram data
        t1 = time.time()
        bins, grayLevels, numGrayLevels = self.getHistogramData(targetVoxels)
        print("Time to calculate histogram: {0} seconds".format(time.time() - t1))

        # First Order Statistics
        if "First-Order Statistics" in feature_categories_keys:
            firstOrderStatistics = FirstOrderStatistics(targetVoxels, bins, numGrayLevels, feature_keys)
            results = firstOrderStatistics.EvaluateFeatures()
            results_storage.update(results)

        # Shape/Size and Morphological Features
        if "Morphology and Shape" in feature_categories_keys:
            # extend padding by one row/column for all 6 directions
            if len(matrix) == 0:
                matrixSA = matrix
                matrixSACoordinates = matrixCoordinates
            else:
                maxDimsSA = tuple(map(operator.add, matrix.shape, ([2, 2, 2])))
                matrixSA, matrixSACoordinates = self.padMatrix(matrix, matrixCoordinates, maxDimsSA, targetVoxels)
            morphologyStatistics = MorphologyStatistics(self._input_ct.GetSpacing(), matrixSA,
                                                                        matrixSACoordinates, targetVoxels, feature_keys)
            results = morphologyStatistics.EvaluateFeatures()
            results_storage.update(results)

        # Texture Features(GLCM)
        if "Texture: GLCM" in feature_categories_keys:
            textureFeaturesGLCM = TextureGLCM(grayLevels, numGrayLevels, matrix, matrixCoordinates,
                                                              targetVoxels, feature_keys, None)
            results = textureFeaturesGLCM.EvaluateFeatures()
            results_storage.update(results)

        # Texture Features(GLRL)
        if "Texture: GLRL" in feature_categories_keys:
            textureFeaturesGLRL = TextureGLRL(grayLevels, numGrayLevels, matrix, matrixCoordinates,
                                                              targetVoxels, feature_keys)
            results = textureFeaturesGLRL.EvaluateFeatures()
            results_storage.update(results)

        # Geometrical Measures
        if "Geometrical Measures" in feature_categories_keys:
            geometricalMeasures = GeometricalMeasures(self._input_ct.GetSpacing(), matrix,
                                                                      matrixCoordinates, targetVoxels, feature_keys)
            results = geometricalMeasures.EvaluateFeatures()
            results_storage.update(results)

        # Renyi Dimensions
        if "Renyi Dimensions" in feature_categories_keys:
            # extend padding to dimension lengths equal to next power of 2
            maxDims = tuple([int(pow(2, math.ceil(np.log2(np.max(matrix.shape)))))] * 3)
            matrixPadded, matrixPaddedCoordinates = self.padMatrix(matrix, matrixCoordinates, maxDims, targetVoxels)
            renyiDimensions = RenyiDimensions(matrixPadded, matrixPaddedCoordinates, feature_keys)
            results = renyiDimensions.EvaluateFeatures()
            results_storage.update(results)

        # Parenchymal Volume
        if "Parenchymal Volume" in feature_categories_keys and subtypes_lm is not None:
            parenchyma_lm_array = sitk.GetArrayFromImage(subtype_lm)
            parenchymalVolume = ParenchymalVolume(parenchyma_lm_array, n_lm_array,
                                                                  self._input_ct.GetSpacing(), feature_keys)
            results = parenchymalVolume.EvaluateFeatures()
            results_storage.update(results)

        # filter for user-queried features only
        results_storage = collections.OrderedDict((k, results_storage[k]) for k in feature_keys)

        return results_storage

    def runAnalysisSphere(self, cid, reference, dist_map, radius, selectedMainFeaturesKeys, selectedFeatureKeys,
                          subtypesWholeVolumeArray=None):
        """ Run the selected features for an sphere of radius r (excluding the nodule itself)
        @param cid: case_id
        @param dist_map: distance map
        @param radius:
        @param subtypesWholeVolumeArray: subtypes volume (only used in parenchyma analysis). np array
        """
        keyName = "{0}_r{1}".format(cid, int(radius))
        sphere_lm_array = self.getSphereLabelMapArray(dist_map, radius)
        
        #sphere_lm = sitk.GetImageFromArray(sphere_lm_array)
        #sphere_lm.CopyInformation(self._input_ct)
        #sitk.WriteImage(sphere_lm,'sphere-'+str(radius)+'.nrrd')
        
        if sphere_lm_array.max() == 0:
            # Nothing to analyze
            results = {}
            for key in selectedFeatureKeys:
                results[key] = 0
            self._analysis_results[keyName,reference] = results
        else:
            self._analysis_results[keyName,reference] = collections.OrderedDict()
            self.runAnalysis(subtypesWholeVolumeArray, sphere_lm_array, self._analysis_results[keyName,reference],
                             selectedMainFeaturesKeys, selectedFeatureKeys)

            print("********* Results for the sphere of radius {0}:".format(radius))
            print(self._analysis_results[keyName,reference])

    def tumorVoxelsAndCoordinates(self, arrayROI, arrayData):
        coordinates = np.where(arrayROI != 0)  # can define specific label values to target or avoid
        values = arrayData[coordinates].astype('int64')
        return values, coordinates

    def paddedTumorMatrixAndCoordinates(self, targetVoxels, targetVoxelsCoordinates):
        if len(targetVoxels) == 0:
            # Nothing to analyze
            empty = np.array([])
            return (empty, (empty, empty, empty))

        ijkMinBounds = np.min(targetVoxelsCoordinates, 1)
        ijkMaxBounds = np.max(targetVoxelsCoordinates, 1)
        matrix = np.zeros(ijkMaxBounds - ijkMinBounds + 1)
        matrixCoordinates = tuple(map(operator.sub, targetVoxelsCoordinates, tuple(ijkMinBounds)))
        matrix[matrixCoordinates] = targetVoxels
        return matrix, matrixCoordinates

    def getHistogramData(self, voxelArray):
        # with np.histogram(), all but the last bin is half-open, so make one extra bin container
        binContainers = np.arange(voxelArray.min(), voxelArray.max() + 2)
        bins = np.histogram(voxelArray, bins=binContainers)[0]  # frequencies
        grayLevels = np.unique(voxelArray)  # discrete gray levels
        numGrayLevels = grayLevels.size
        return bins, grayLevels, numGrayLevels

    def padMatrix(self, a, matrixCoordinates, dims, voxelArray):
        # pads matrix 'a' with zeros and resizes 'a' to a cube with dimensions increased to the next greatest power of 2
        # np version 1.7 has np.pad function

        # center coordinates onto padded matrix    # consider padding with NaN or eps = np.spacing(1)
        pad = tuple(map(operator.div, tuple(map(operator.sub, dims, a.shape)), ([2, 2, 2])))
        matrixCoordinatesPadded = tuple(map(operator.add, matrixCoordinates, pad))
        matrix2 = np.zeros(dims)
        matrix2[matrixCoordinatesPadded] = voxelArray
        return matrix2, matrixCoordinatesPadded

    def getPredefinedSpheresDict(self, i_ct):
        """Get predefined spheres """
        spheresDict = dict()
        spheresDict[self.WORKING_MODE_HUMAN] = (15, 20, 25)  # Humans
        spheresDict[self.WORKING_MODE_SMALL_ANIMAL] = (1.5, 2, 2.5)  # Mouse

        return spheresDict[self.getWorkingMode(i_ct)]

    def getWorkingMode(self, i_ct):
        """
        Get the right working mode for this volume based on the size
        @param i_ct:
        @return: self.WORKING_MODE_HUMAN or self.WORKING_MODE_SMALL_ANIMAL
        """
        size = i_ct.GetSpacing()[0] * i_ct.GetImageData().GetDimensions()[0]
        return self.WORKING_MODE_HUMAN if size >= 100 else self.WORKING_MODE_SMALL_ANIMAL

    def compute_centroid(self, np_array, labelId=1):
        """ Calculate the coordinates of a centroid for a concrete labelId (default=1)
        :param np_array: np array
        :param labelId: label id (default = 1)
        :return: np array with the coordinates (int format)
        """
        mean = np.mean(np.where(np_array == labelId), axis=1)
        return np.asarray(np.round(mean, 0), np.int)

    def vtk_np_coordinate(self, vtk_coordinate):
        """ Adapt a coordinate in VTK to a np array format handled by VTK (ie in a reversed order)
        :param itk_coordinate: coordinate in VTK (xyz)
        :return: coordinate in np (zyx)
        """
        l = list(vtk_coordinate)
        l.reverse()
        return l

    def np_itk_coordinate(self, np_coordinate, convert_to_int=True):
        """ Adapt a coordinate in np to a ITK format (ie in a reversed order and converted to int type)
        :param np_coordinate: coordinate in np (zyx)
        :param convert_to_int: convert the coordinate to int type, needed for SimpleITK image coordinates
        :return: coordinate in ITK (xyz)
        """
        if convert_to_int:
            return [int(np_coordinate[2]), int(np_coordinate[1]), int(np_coordinate[0])]
        return [np_coordinate[2], np_coordinate[1], np_coordinate[0]]

    def getCurrentDistanceMapFromCentroid(self,whole_lm):
        """ Calculate the distance map to the centroid for the current labelmap volume.
        To that end, we have to calculate first the centroid.
        Please note the results could be cached
        @return:
        """
        nodule_lm_array = sitk.GetArrayFromImage(self._nodule_lm)
        centroid = self.compute_centroid(nodule_lm_array)
        # Calculate the distance map for the specified origin
        # Get the dimensions of the volume in ZYX coords
        i_ct_array = sitk.GetArrayFromImage(self._input_ct)
        dims = i_ct_array.shape
        # Speed map (all ones because the growth will be constant).
        # The dimensions are reversed because we want the format in ZYX coordinates
        input = np.ones(dims, np.int32)

        sitkImage = sitk.GetImageFromArray(input)
        sitkImage.SetSpacing(self._input_ct.GetSpacing())
        fastMarchingFilter = sitk.FastMarchingImageFilter()
        fastMarchingFilter.SetStoppingValue(self.MAX_TUMOR_RADIUS)
        # Reverse the coordinate of the centroid
        seeds = [self.np_itk_coordinate(centroid)]
        fastMarchingFilter.SetTrialPoints(seeds)
        output = fastMarchingFilter.Execute(sitkImage)
        output_array=sitk.GetArrayFromImage(output)

        # Make sure that DM does not overextend the whole lung
        # We did not do this setting up the speed image to zero because sometimes the centroid can fall
        # outside the lung field, and we will kill the evolution.
        # We rather set the distance map to an imposible value (-1) outside the lung.
        if whole_lm is not None:
            whole_lm_array = sitk.GetArrayFromImage(whole_lm)
            #Set speed image to 0 outside the lung
            output_array[whole_lm_array==0]=-1

        # self._current_distance_maps[cid] = sitk.GetArrayFromImage(output)

        return output_array


    def getCurrentDistanceMapFromNodule(self,whole_lm):
        """ Calculate the distance map to the surface of the nodule.
        Please note the results could be cached
        @return:
        """
    
        #Set Distance map filter
        output=sitk.SignedMaurerDistanceMap(self._nodule_lm,False,False,True)
        output_array=sitk.GetArrayFromImage(output)
        
        if whole_lm is not None:
            whole_lm_array = sitk.GetArrayFromImage(whole_lm)
            #Set speed image to 0 outside the lung
            output_array[whole_lm_array==0]=-1
        
        # self._current_distance_maps[cid] = sitk.GetArrayFromImage(output)
        
        return output_array

    def getSphereLabelMapArray(self, dm, radius):
        """ Get a labelmap np array that contains a sphere centered in the nodule centroid, with radius "radius" and that
        EXCLUDES the nodule itself.
        If the results are not cached, this method creates the volume and calculates the labelmap
        @param radius: radius of the sphere
        @return: labelmap array for a sphere of this radius
        """
        # If the sphere was already calculated, return the results
        # name = "SphereLabelmap_r{0}".format(radius)
        # Try to get first the node from the subject hierarchy tree
        # Otherwise, Init with the current segmented nodule labelmap
        # Create and save the labelmap in the Subject hierarchy
        nodule_lm_array = sitk.GetArrayFromImage(self._nodule_lm)
        sphere_lm_array = sitk.GetArrayFromImage(self._nodule_lm)
        # Mask with the voxels that are inside the radius of the sphere
        # dm = self._current_distance_maps[cid]
        #Make sure that we analyze dm>0 to exclude areas outside the lung with dm=-1
        sphere_lm_array[(dm <= radius) & ( dm > 0 )] = 1
        # Exclude the nodule
        sphere_lm_array[nodule_lm_array == 1] = 0
        return sphere_lm_array

    def saveReport(self, cid):
        """ Save the current values in a persistent csv file
        """
        keyName = cid
        radius = ''
        reference="Interior"
        
        self.saveBasicData(keyName,cid,radius,reference)
        self.saveCurrentValues(self._analysis_results[keyName,reference])

        # Get all the spheres for this nodule
        analyzedSpheres=dict()
        analyzedSpheres["Centroid"]=self._centroid_analyzedSpheres
        analyzedSpheres["Boundary"]=self._boundary_analyzedSpheres

        for reference in ["Centroid","Boundary"]:
            #print analyzedSpheres[reference]
            for rad in analyzedSpheres[reference]:
                keyName = "{}_r{}".format(cid, int(rad))
                radius = rad
                print ("Saving basic data "+str(radius))
                self.saveBasicData(keyName,cid,radius,reference)
                self.saveCurrentValues(self._analysis_results[keyName,reference])

    def saveBasicData(self, keyName,cid,rad,reference):
        date = time.strftime("%Y/%m/%d %H:%M:%S")
        self._analysis_results[keyName,reference]["Case ID"] = cid
        self._analysis_results[keyName,reference]["Sphere Radius"] = rad
        self._analysis_results[keyName,reference]["Sphere Reference"] = reference
        self._analysis_results[keyName,reference]["Nodule Number"] = self._nodule_id
        self._analysis_results[keyName,reference]["Date"] = date
        self._analysis_results[keyName,reference]["Lesion Type"] = self._lesion_type
        if len(self._seed_point) > 0:
            self._analysis_results[keyName,reference]["Seeds (LPS)"] = self._seed_point
        else:
            self._analysis_results[keyName,reference]["Seeds (LPS)"] = ''
        if self._segmentation_threshold is not None:
            self._analysis_results[keyName,reference]["Threshold"] = self._segmentation_threshold
        else:
            self._analysis_results[keyName,reference]["Threshold"] = ''

    def saveCurrentValues(self, analysis_results):
        """ Save a new row of information in the current csv file that stores the data  (from a dictionary of items)
        :param kwargs: dictionary of values
        """
        # Check that we have all the "columns"
        storedColumnNames = ["Case ID", "Sphere Radius", "Sphere Reference", "Nodule Number", "Date", "Lesion Type", "Seeds (LPS)", "Threshold"]
        # Create a single features list with all the "child" features
        storedColumnNames.extend(itertools.chain.from_iterable(self._all_features.itervalues()))

        orderedColumns = []
        # Always add a timestamp as the first value
        # orderedColumns.append(time.strftime("%Y/%m/%d %H:%M:%S"))
        for column in storedColumnNames:
            if analysis_results.has_key(column):
                orderedColumns.append(analysis_results[column])
            else:
                orderedColumns.append('')

        with open(self._csv_file_path, 'a+b') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(orderedColumns)


def ras_to_lps(coords):
    """ Convert from RAS to LPS or viceversa (it is just flipping the first axis)
    :return: list of 3 coordinates
    """
    lps_to_ras_matrix = vtk.vtkMatrix4x4()
    lps_to_ras_matrix.SetElement(0, 0, -1)
    lps_to_ras_matrix.SetElement(1, 1, -1)

    cl = list(coords)
    cl.append(1)

    return list(lps_to_ras_matrix.MultiplyPoint(cl)[:-1])


def run_lung_segmentation(i_ct_filename, o_lm):
    """ Run the nodule segmentation through a CLI
    """
    tmpCommand = "GeneratePartialLungLabelMap --ict %(in)s --olm %(out)s"
    tmpCommand = tmpCommand % {'in': i_ct_filename, 'out': o_lm}
    # tmpCommand = os.path.join(path['CIP_PATH'], tmpCommand)
    subprocess.call(tmpCommand, shell=True)


def write_csv_first_row(csv_file_path, feature_classes):
    column_names = ["Case ID", "Sphere Radius", "Sphere Reference", "Nodule Number", "Date", "Lesion Type", "Seeds (LPS)", "Threshold"]
    column_names.extend(itertools.chain.from_iterable(feature_classes.itervalues()))
    with open(csv_file_path, 'w+b') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)


def get_all_features():
    feature_classes = collections.OrderedDict()
    feature_classes["First-Order Statistics"] = ["Voxel Count", "Gray Levels", "Energy", "Entropy",
                                                 "Minimum Intensity", "Maximum Intensity", "Mean Intensity",
                                                 "Median Intensity", "Range", "Mean Deviation",
                                                 "Root Mean Square", "Standard Deviation",
                                                 "Ventilation Heterogeneity", "Skewness", "Kurtosis",
                                                 "Variance", "Uniformity","LAA950Perc","LAA910Perc","LAA856Perc"]

    feature_classes["Morphology and Shape"] = ["Volume mm^3", "Volume cc", "Surface Area mm^2",
                                               "Surface:Volume Ratio", "Compactness 1", "Compactness 2",
                                               "Maximum 3D Diameter", "Spherical Disproportion",
                                               "Sphericity"]

    feature_classes["Texture: GLCM"] = ["Autocorrelation", "Cluster Prominence", "Cluster Shade",
                                        "Cluster Tendency", "Contrast", "Correlation",
                                        "Difference Entropy", "Dissimilarity", "Energy (GLCM)", "Entropy(GLCM)",
                                        "Homogeneity 1", "Homogeneity 2", "IMC1", "IDMN", "IDN", "Inverse Variance",
                                        "Maximum Probability", "Sum Average", "Sum Entropy",
                                        "Sum Variance", "Variance (GLCM)"]  # IMC2 missing

    feature_classes["Texture: GLRL"] = ["SRE", "LRE", "GLN", "RLN", "RP", "LGLRE", "HGLRE", "SRLGLE",
                                        "SRHGLE", "LRLGLE", "LRHGLE"]

    feature_classes["Geometrical Measures"] = ["Extruded Surface Area", "Extruded Volume",
                                               "Extruded Surface:Volume Ratio"]

    feature_classes["Renyi Dimensions"] = ["Box-Counting Dimension", "Information Dimension", "Correlation Dimension"]
    feature_classes["Parenchymal Volume"] = ParenchymalVolume.getAllEmphysemaDescriptions()
    return feature_classes


def get_nodule_information(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    nodules_id = []
    nodules_type = []
    nodules_seed = []

    coord_syst = root.find('CoordinateSystem').text
    cc = ChestConventions()
    #NoduleChestTypes=[86,87,88]
    NoduleChestTypes=['Nodule','BenignNodule','MalignantNodule']

    for n in root.findall('Point'):
        chesttype=int(n.find('ChestType').text)
        chesttype_name=cc.GetChestTypeNameFromValue(chesttype)
        if chesttype_name in NoduleChestTypes:
          n_id = n.find('Id').text
          nodules_id.append(n_id)
          #t = n.find('Description').text
          #nodules_type.append(t)
          nodules_type.append(chesttype_name)
          coordinate = n.find('Coordinate')
          seed = []
          for s in coordinate.findall('value'):
            seed.append(s.text)
          nodules_seed.append(seed)

    return coord_syst, nodules_id, nodules_type, nodules_seed


if __name__ == "__main__":
    desc = """This module allows to segment benign nodules and tumors in the lung.
            Besides, it analyzes a lot of different features inside the nodule and in its surroundings,
            in concentric spheres of different radius centered in the centroid of the nodule"""

    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                        help='Input CT file', dest='in_ct', metavar='<string>',
                        default=None)
    parser.add_option('--xml',
                        help='XML file containing nodule information for the input ct.', dest='xml_file',
                        metavar='<string>', default=None)
    parser.add_option('--cid',
                      help='Case ID to use otherwise it is guess from filename (not recommended)',dest='cid',
                      metavar='<string>', default=None)
    parser.add_option('--coords',
                      help='coordinates of the nodule if xml is not provided (LPS coordinates)',dest='coords',
                      metavar='<string>', default=None)
    parser.add_option('--type',
                          help='type of nodule if xml is not provided',dest='type',
                          metavar='<string>', default=None)
    parser.add_option('--n_lm',
                        help='Nodule labelmap. If labelmap exists, it will be used for \
                        analysis. Otherwise, nodule will be segmented first.', dest='n_lm',
                        metavar='<string>', default=None)
    parser.add_option('--max_rad',
                        help='Maximum radius (mm) for the lesion. Recommended: 30 mm \
                        for humans and 3 mm for small animals',
                        dest='max_rad', metavar='<float>', default=30.0)
    parser.add_option('--th',
                        help='Threshold value for nodule segmentation. All the voxels above the threshold will be \
                              considered nodule)',
                        dest='segm_th', metavar='<float>', default=None)
    parser.add_option('--par_lm',
                        help='Partial lung labelmap. If labelmap exists, it will be used for parenchyma analysis. \
                        Otherwise, labelmap will be created first.',
                        dest='par_lm', metavar='<string>', default=None)
    parser.add_option('--subtypes_lm',
                        help='Subtypes labelmap to use for parenchyma analysis.',
                        dest='subtypes_lm', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                        help='CSV file to save nodule analysis.', dest='csv_file', metavar='<string>',
                        default=None)
    parser.add_option('--compute_all',
                        help='Set this flag to compute all features of all classes. If not setting this flag, \
                        select features to be computed.', dest='compute_all', action='store_true')
    parser.add_option('--fos_feat',
                        help='First Order Statistics features. For computation of \
                        all fos features indicate all.',
                        dest='fos_features', metavar='<string>', default=None)
    parser.add_option('--ms_feat',
                        help='Morphology and Shape features. For computation of \
                        all ms features indicate all.',
                        dest='ms_features', metavar='<string>', default=None)
    parser.add_option('--glcm_feat',
                        help='Gray-Level Co-ocurrence Matrices features. For computation of \
                        all glcm features indicate all.',
                        dest='glcm_features', metavar='<string>', default=None)
    parser.add_option('--glrl_feat',
                        help='Gray-Level Run Length features. For computation of \
                        all glrl features indicate all.',
                        dest='glrl_features', metavar='<string>', default=None)
    parser.add_option('--renyi_dim',
                        help='Renyi Dimensions. For computation of all renyi dimensions indicate all.',
                        dest='renyi_dimensions', metavar='<string>', default=None)
    parser.add_option('--geom_meas',
                        help='Geometrical Measures. For computation of all renyi dimensions indicate all.',
                        dest='geom_measures', metavar='<string>', default=None)
    parser.add_option('--par_feat',
                        help='Parenchymal volume features. For computation of \
                                  all features indicate all.',
                        dest='par_features', metavar='<string>', default=None)
    parser.add_option('--centroid_sphere_rad',
                        help='Radius(es) for Sphere computation computed from the centroid of the nodule.', metavar='<float>', dest='centroid_sphere_rads',
                        default=None)
    parser.add_option('--boundary_sphere_rad',
                        help='Radius(es) for Sphere computation computed from the boundary of the nodule surface.', metavar='<float>', dest='boundary_sphere_rads',
                        default=None)
    parser.add_option('--tmp',
                        help='Temp directory for saving computed labelmaps.', metavar='<string>',
                        dest='tmp_dir', default=None)

    (options, args) = parser.parse_args()

    input_ct = sitk.ReadImage(options.in_ct)
    fileparts = os.path.splitext(options.in_ct)
    
    if options.cid is None:
        case_id = fileparts[0].split('/')[-1:][0]
    else:
        case_id = options.cid

    n_lm_names = options.n_lm
    if n_lm_names is not None:
        n_lm_names = [str(lm) for lm in str.split(options.n_lm, ',')]

    if options.xml_file is not None:
        coord_system, ids, types, seeds = get_nodule_information(options.xml_file)
    else:
        if options.coords is not None:
            no_xml_coords = [float(t) for t in str.split(options.coords, ',')]
        else:
            no_xml_coords = [0,0,0]

        if options.type is not None:
            no_xml_type = options.type
        else:
            no_xml_type = "Undefined"
    
    
    #max_rad = [float(r) for r in str.split(options.max_rad, ',')]
    max_rad = float(options.max_rad)

    segm_th = options.segm_th
    if segm_th is not None:
        segm_th = [float(t) for t in str.split(options.segm_th, ',')]
    c_sph_rads = options.centroid_sphere_rads
    if c_sph_rads is not None:
        c_sph_rads = [float(sr) for sr in str.split(options.centroid_sphere_rads, ',')]

    b_sph_rads = options.boundary_sphere_rads
    if b_sph_rads is not None:
        b_sph_rads = [float(sr) for sr in str.split(options.boundary_sphere_rads, ',')]


    all_feature_classes = get_all_features()
    feature_classes = collections.OrderedDict()

    tmp_dir = options.tmp_dir
    if tmp_dir is not None and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    elif tmp_dir is None and (options.par_lm is None or options.n_lm is None):
        #Auto create tmp_dir only if we need it to store par_lm or n_lm
        tmp_dir = os.path.join(os.getcwd(), 'LabelMaps')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    parenchyma_lm = None

    par_lm_filename = options.par_lm
    if par_lm_filename is None:
        par_lm_filename = tmp_dir + '/' + case_id + '_partialLungLabelMap.nrrd'
        if not os.path.exists(par_lm_filename):
            run_lung_segmentation(options.in_ct, par_lm_filename)

        parenchyma_lm = sitk.ReadImage(par_lm_filename)
        parenchyma_lm_array = sitk.GetArrayFromImage(parenchyma_lm)
        parenchyma_lm_array[parenchyma_lm_array > 0] = 1
        parenchyma_lm = sitk.GetImageFromArray(parenchyma_lm_array)
        parenchyma_lm.SetSpacing(input_ct.GetSpacing())
        parenchyma_lm.SetOrigin(input_ct.GetOrigin())
    else:
        parenchyma_lm = sitk.ReadImage(par_lm_filename)

    subtypes_lm = None
    if options.compute_all:
        feature_classes = all_feature_classes
        subtypes_lm_filename = options.subtypes_lm
        if subtypes_lm_filename is not None:
            subtypes_lm = sitk.ReadImage(par_lm_filename)
        else:
            print ("Subtype image is necessary to compute all phenotypes")

    else:
        if options.fos_features == 'all':
            feature_classes["First-Order Statistics"] = all_feature_classes["First-Order Statistics"]
        elif options.fos_features is not None:
            feature_classes["First-Order Statistics"] = [ff for ff in str.split(options.fos_features, ',')]

        if options.ms_features == 'all':
            feature_classes["Morphology and Shape"] = all_feature_classes["Morphology and Shape"]
        elif options.ms_features is not None:
            feature_classes["Morphology and Shape"] = [ff for ff in str.split(options.ms_features, ',')]

        if options.glcm_features == 'all':
            feature_classes["Texture: GLCM"] = all_feature_classes["Texture: GLCM"]
        elif options.glcm_features is not None:
            feature_classes["Texture: GLCM"] = [ff for ff in str.split(options.glcm_features, ',')]

        if options.glrl_features == 'all':
            feature_classes["Texture: GLRL"] = all_feature_classes["Texture: GLRL"]
        elif options.glrl_features is not None:
            feature_classes["Texture: GLRL"] = [ff for ff in str.split(options.glrl_features, ',')]

        if options.geom_measures == 'all':
            feature_classes["Geometrical Measures"] = all_feature_classes["Geometrical Measures"]
        elif options.geom_measures is not None:
            feature_classes["Geometrical Measures"] = [ff for ff in str.split(options.geom_measures, ',')]

        if options.renyi_dimensions == 'all':
            feature_classes["Renyi Dimensions"] = all_feature_classes["Renyi Dimensions"]
        elif options.renyi_dimensions is not None:
            feature_classes["Renyi Dimensions"] = [ff for ff in str.split(options.renyi_dimensions, ',')]

        if options.par_features is not None:
            subtypes_lm_filename = options.subtypes_lm
            if subtypes_lm_filename is not None:
                subtypes_lm = sitk.ReadImage(par_lm_filename)
                if options.par_features == 'all':
                    feature_classes["Parenchymal Volume"] = all_feature_classes["Parenchymal Volume"]
                elif options.par_features is not None:
                    feature_classes["Parenchymal Volume"] = [ff for ff in str.split(options.par_features, ',')]
            else:
                print ("Parenchymal features will not be computed because subtype labelmap is not provided")


    nodule_lm_list=[]
    nodule_id_list=[]
    nodule_type_list=[]
    nodule_coord_list=[]
    nodule_th_list=[]
    if options.xml_file is not None:
        #Parse the xml and create nodule segmentation for each point
        for i in range(len(ids)):
            seed_point = [float(s) for s in seeds[i]]
            if coord_system == 'RAS':
                seed_point = ras_to_lps(seed_point)

            #lesion_type = types[i][2:]
            lesion_type = types[i]
            nodule_id = ids[i]
            max_radius = max_rad
            if segm_th is not None:
                segm_threshold = segm_th[i]
            else:
                segm_threshold = None

            if n_lm_names is None:
                n_lm_filename = tmp_dir + '/' + case_id + '_noduleLabelMap.nrrd'
            elif n_lm_names[i] is None:
                n_lm_filename = tmp_dir + '/' + case_id + '_noduleLabelMap.nrrd'
            else:
                n_lm_filename = n_lm_names[i]

            if not os.path.exists(n_lm_filename):
                nodule_segmenter = NoduleSegmenter(input_ct, options.in_ct, max_radius, seed_point,
                                                   n_lm_filename, segm_threshold)
                nodule_segmenter.segment_nodule()
            nodule_lm_list.append(n_lm_filename)
            nodule_id_list.append(nodule_id)
            nodule_type_list.append(lesion_type)
            nodule_coord_list.append(seed_point)
            nodule_th_list.append(segm_threshold)
    else:
        #Define naive default values. This is just a fall back in case we only get a nodule labelmap
        if segm_th is None:
            segm_th=[0.5]
        
        if n_lm_names is None:
            n_lm_filename = tmp_dir + '/' + case_id + '_noduleLabelMap.nrrd'
            nodule_segmenter = NoduleSegmenter(input_ct, options.in_ct, max_rad, no_xml_coords,
                                           n_lm_filename, segm_th[0])
            nodule_segmenter.segment_nodule()
            n_lm_names=[n_lm_filename]

        for i in range(len(n_lm_names)):
            nodule_lm_list.append(n_lm_names[i])
            nodule_id_list.append(i)
            nodule_type_list.append(no_xml_type)
            nodule_coord_list.append(no_xml_coords)
            nodule_th_list.append(segm_th[0])

#if not os.path.exists(options.csv_file):
    write_csv_first_row(options.csv_file, all_feature_classes)

    for i in range(len(nodule_lm_list)):
        
        n_lm_filename=nodule_lm_list[i]
        nodule_id=nodule_id_list[i]
        lesion_type=nodule_type_list[i]
        seed_point=nodule_coord_list[i]
        segm_threshold=nodule_th_list[i]

        nodule_lm = sitk.ReadImage(n_lm_filename)

#        for j in range (len(sph_rads)):
#            if sph_rads is not None:
#                sphere_rad = sph_rads[j]
#            else:
#                sphere_rad = 0.0


        ns = NodulePhenotypes(input_ct, nodule_lm, nodule_id, seed_point, lesion_type, segm_threshold, c_sph_rads,b_sph_rads,
                             feature_classes, options.csv_file, all_feature_classes,subtypes_lm=subtypes_lm)
        ns.execute(case_id, whole_lm=parenchyma_lm)
