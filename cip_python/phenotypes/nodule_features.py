import vtk
import string
import numpy
import math
import operator
import collections
import time
import numpy as np

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

        self.parameterValues = parameterValues
        self.bins = bins
        self.grayLevels = grayLevels
        self.keys = set(allKeys).intersection(self.firstOrderStatistics.keys())

    def voxelCount(self, parameterArray):
        return (parameterArray.size)

    def grayLevelCount(self, grayLevels):
        return (grayLevels)

    def energyValue(self, parameterArray):
        return (numpy.sum(parameterArray ** 2))

    def entropyValue(self, bins):
        return (numpy.sum(bins * numpy.where(bins != 0, numpy.log2(bins), 0)))

    def minIntensity(self, parameterArray):
        return (numpy.min(parameterArray))

    def maxIntensity(self, parameterArray):
        return (numpy.max(parameterArray))

    def meanIntensity(self, parameterArray):
        return (numpy.mean(parameterArray))

    def medianIntensity(self, parameterArray):
        return (numpy.median(parameterArray))

    def rangeIntensity(self, parameterArray):
        return (numpy.max(parameterArray) - numpy.min(parameterArray))

    def meanDeviation(self, parameterArray):
        return (numpy.mean(numpy.absolute((numpy.mean(parameterArray) - parameterArray))))

    def rootMeanSquared(self, parameterArray):
        return (((numpy.sum(parameterArray ** 2)) / (parameterArray.size)) ** (1 / 2.0))

    def standardDeviation(self, parameterArray):
        return (numpy.std(parameterArray))

    def ventilationHeterogeneity(self, parameterArray):
        # Keep just the points that are in the range (-1000, 0]
        arr = parameterArray[((parameterArray > -1000) & (parameterArray <= 0))]
        # Convert to float to apply the formula
        arr = arr.astype(numpy.float)
        # Apply formula
        arr = -arr / (arr + 1000)
        arr **= (1/3.0)
        return arr.std()

    def _moment(self, a, moment=1, axis=0):
        # Modified from SciPy module
        if moment == 1:
            return numpy.float64(0.0)
        else:
            mn = numpy.expand_dims(numpy.mean(a, axis), axis)
            s = numpy.power((a - mn), moment)
            return numpy.mean(s, axis)

    def skewnessValue(self, a, axis=0):
        # Modified from SciPy module
        # Computes the skewness of a dataset

        m2 = self._moment(a, 2, axis)
        m3 = self._moment(a, 3, axis)

        # Control Flow: if m2==0 then vals = 0; else vals = m3/m2**1.5
        zero = (m2 == 0)
        vals = numpy.where(zero, 0, m3 / m2 ** 1.5)

        if vals.ndim == 0:
            return vals.item()
        return vals

    def kurtosisValue(self, a, axis=0, fisher=True):
        # Modified from SciPy module

        m2 = self._moment(a, 2, axis)
        m4 = self._moment(a, 4, axis)
        zero = (m2 == 0)

        # Set Floating-Point Error Handling
        olderr = numpy.seterr(all='ignore')
        try:
            vals = numpy.where(zero, 0, m4 / m2 ** 2.0)
        finally:
            numpy.seterr(**olderr)
        if vals.ndim == 0:
            vals = vals.item()  # array scalar

        if fisher:
            return vals - 3
        else:
            return vals

    def varianceValue(self, parameterArray):
        return (numpy.std(parameterArray) ** 2)

    def uniformityValue(self, bins):
        return (numpy.sum(bins ** 2))

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
            for l in xrange(l_slice.start, l_slice.stop):
                fxy = numpy.array([extrudedMatrix[i + 1, j, k, l], extrudedMatrix[i - 1, j, k, l]]) == 0
                fyz = numpy.array([extrudedMatrix[i, j + 1, k, l], extrudedMatrix[i, j - 1, k, l]]) == 0
                fxz = numpy.array([extrudedMatrix[i, j, k + 1, l], extrudedMatrix[i, j, k - 1, l]]) == 0
                f4d = numpy.array([extrudedMatrix[i, j, k, l + 1], extrudedMatrix[i, j, k, l - 1]]) == 0

                extrudedElementSurface = (numpy.sum(fxz) * xz) + (numpy.sum(fyz) * yz) + (numpy.sum(fxy) * xy) + (
                numpy.sum(f4d) * fourD)
                extrudedSurfaceArea += extrudedElementSurface
        return (extrudedSurfaceArea)

    def extrudedVolume(self, extrudedMatrix, extrudedMatrixCoordinates, cubicMMPerVoxel):
        extrudedElementsSize = extrudedMatrix[numpy.where(extrudedMatrix == 1)].size
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

        parameterValues = numpy.abs(parameterValues)

        # maximum intensity/parameter value appended as shape of 4th dimension
        extrudedShape = parameterMatrix.shape + (numpy.max(parameterValues),)

        # pad shape by 1 unit in all 8 directions
        extrudedShape = tuple(map(operator.add, extrudedShape, [2, 2, 2, 2]))

        extrudedMatrix = numpy.zeros(extrudedShape)
        extrudedMatrixCoordinates = tuple(map(operator.add, parameterMatrixCoordinates, ([1, 1, 1]))) + (
        numpy.array([slice(1, value + 1) for value in parameterValues]),)
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
        for voxel in xrange(0, matrixSAValues.size):
            i, j, k = matrixSACoordinates[0][voxel], matrixSACoordinates[1][voxel], matrixSACoordinates[2][voxel]
            fxy = (numpy.array([a[i + 1, j, k], a[i - 1, j, k]]) == 0)  # evaluate to 1 if true, 0 if false
            fyz = (numpy.array([a[i, j + 1, k], a[i, j - 1, k]]) == 0)  # evaluate to 1 if true, 0 if false
            fxz = (numpy.array([a[i, j, k + 1], a[i, j, k - 1]]) == 0)  # evaluate to 1 if true, 0 if false
            surface = (numpy.sum(fxz) * xz) + (numpy.sum(fyz) * yz) + (numpy.sum(fxy) * xy)
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

        minBounds = numpy.array(
            [numpy.min(matrixSACoordinates[0]), numpy.min(matrixSACoordinates[1]), numpy.min(matrixSACoordinates[2])])
        maxBounds = numpy.array(
            [numpy.max(matrixSACoordinates[0]), numpy.max(matrixSACoordinates[1]), numpy.max(matrixSACoordinates[2])])

        a = numpy.array(zip(*matrixSACoordinates))
        edgeVoxelsMinCoords = numpy.vstack(
            [a[a[:, 0] == minBounds[0]], a[a[:, 1] == minBounds[1]], a[a[:, 2] == minBounds[2]]]) * [z, y, x]
        edgeVoxelsMaxCoords = numpy.vstack(
            [(a[a[:, 0] == maxBounds[0]] + 1), (a[a[:, 1] == maxBounds[1]] + 1), (a[a[:, 2] == maxBounds[2]] + 1)]) * [
                                  z, y, x]

        maxDiameter = 1
        for voxel1 in edgeVoxelsMaxCoords:
            for voxel2 in edgeVoxelsMinCoords:
                voxelDistance = numpy.sqrt(numpy.sum((voxel2 - voxel1) ** 2))
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

        # exception for numpy.sum(c) = 0?
        c = c / float(numpy.sum(c))
        maxDim = c.shape[0]
        p = int(numpy.log2(maxDim))
        n = numpy.zeros(p + 1)
        eps = numpy.spacing(1)

        # Initialize N(s) value at the finest/voxel-level scale
        if (q == 1):
            n[p] = numpy.sum(c[matrixCoordinatesPadded] * numpy.log(1 / (c[matrixCoordinatesPadded] + eps)))
        else:
            n[p] = numpy.sum(c[matrixCoordinatesPadded] ** q)

        for g in xrange(p - 1, -1, -1):
            siz = 2 ** (p - g)
            siz2 = round(siz / 2)
            for i in xrange(0, maxDim - siz + 1, siz):
                for j in xrange(0, maxDim - siz + 1, siz):
                    for k in xrange(0, maxDim - siz + 1, siz):
                        box = numpy.array([c[i, j, k], c[i + siz2, j, k], c[i, j + siz2, k], c[i + siz2, j + siz2, k],
                                           c[i, j, k + siz2], c[i + siz2, j, k + siz2], c[i, j + siz2, k + siz2],
                                           c[i + siz2, j + siz2, k + siz2]])
                        c[i, j, k] = numpy.any(box != 0) if (q == 0) else numpy.sum(box) ** q
                    if self.checkStopProcessFunction is not None:
                        self.checkStopProcessFunction()
                        # print (i, j, k, '                ', c[i,j,k])
            pi = c[0:(maxDim - siz + 1):siz, 0:(maxDim - siz + 1):siz, 0:(maxDim - siz + 1):siz]
            if (q == 1):
                n[g] = numpy.sum(pi * numpy.log(1 / (pi + eps)))
            else:
                n[g] = numpy.sum(pi)
                # print ('p, g, siz, siz2', p, g, siz, siz2, '         n[g]: ', n[g])

        r = numpy.log(2.0 ** (numpy.arange(p + 1)))  # log(1/scale)
        scaleMatrix = numpy.array([r, numpy.ones(p + 1)])
        # print ('n(s): ', n)
        # print ('log (1/s): ', r)

        if (q != 1):
            n = (1 / float(1 - q)) * numpy.log(n)
        renyiDimension = numpy.linalg.lstsq(scaleMatrix.T, n)[0][0]

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
        # make distance an optional parameter, as in: distances = numpy.arange(parameter)
        distances = numpy.array([1])
        directions = 26
        self.P_glcm = numpy.zeros((self.Ng, self.Ng, distances.size, directions))
        t1 = time.time()
        self.P_glcm = self.calculate_glcm(self.grayLevels, self.parameterMatrix, self.parameterMatrixCoordinates,
                                          distances, directions, self.Ng, self.P_glcm)
        if printTiming:
            print("- Time to calculate glmc matrix: {0} secs".format(time.time() - t1))
        # make each GLCM symmetric an optional parameter
        # if symmetric:
        # Pt = numpy.transpose(P, (1, 0, 2, 3))
        # P = P + Pt

        ##Calculate GLCM Coefficients
        self.ivector = numpy.arange(1, self.Ng + 1)  # shape = (self.Ng, distances.size, directions)
        self.jvector = numpy.arange(1, self.Ng + 1)  # shape = (self.Ng, distances.size, directions)
        self.eps = numpy.spacing(1)

        self.prodMatrix = numpy.multiply.outer(self.ivector, self.jvector)  # shape = (self.Ng, self.Ng)
        self.sumMatrix = numpy.add.outer(self.ivector, self.jvector)  # shape = (self.Ng, self.Ng)
        self.diffMatrix = numpy.absolute(numpy.subtract.outer(self.ivector, self.jvector))  # shape = (self.Ng, self.Ng)
        self.kValuesSum = numpy.arange(2, (self.Ng * 2) + 1)  # shape = (2*self.Ng-1)
        self.kValuesDiff = numpy.arange(0, self.Ng)  # shape = (self.Ng-1)

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
        self.pxAddy = numpy.array([numpy.sum(self.P_glcm[self.sumMatrix == k], 0) for k in self.kValuesSum])
        # shape = (self.Ng, distances.size, directions)
        self.pxSuby = numpy.array([numpy.sum(self.P_glcm[self.diffMatrix == k], 0) for k in self.kValuesDiff])

        # entropy of self.px #shape = (distances.size, directions)
        self.HX = (-1) * numpy.sum((self.px * numpy.where(self.px != 0, numpy.log2(self.px), numpy.log2(self.eps))), 0)
        # entropy of py #shape = (distances.size, directions)
        self.HY = (-1) * numpy.sum((self.py * numpy.where(self.py != 0, numpy.log2(self.py), numpy.log2(self.eps))), 0)
        # shape = (distances.size, directions)
        self.HXY = (-1) * numpy.sum(
            numpy.sum((self.P_glcm * numpy.where(self.P_glcm != 0, numpy.log2(self.P_glcm), numpy.log2(self.eps))), 0),
            0)

        self.pxy = numpy.zeros(self.P_glcm.shape)  # shape = (self.Ng, self.Ng, distances.size, directions)
        for a in xrange(directions):
            for g in xrange(distances.size):
                self.pxy[:, :, g, a] = numpy.multiply.outer(self.px[:, g, a], self.py[:, g, a])

        self.HXY1 = (-1) * numpy.sum(
            numpy.sum((self.P_glcm * numpy.where(self.pxy != 0, numpy.log2(self.pxy), numpy.log2(self.eps))), 0),
            0)  # shape = (distances.size, directions)
        self.HXY2 = (-1) * numpy.sum(
            numpy.sum((self.pxy * numpy.where(self.pxy != 0, numpy.log2(self.pxy), numpy.log2(self.eps))), 0),
            0)  # shape = (distances.size, directions)
        if printTiming:
            print("- Time to calculate total glmc coefficients: {0} secs".format(time.time() - t1))

    def autocorrelationGLCM(self, P_glcm, prodMatrix, meanFlag=True):
        ac = numpy.sum(numpy.sum(P_glcm * prodMatrix[:, :, None, None], 0), 0)
        if meanFlag:
            return (ac.mean())
        else:
            return ac

    def clusterProminenceGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        cp = numpy.sum(
            numpy.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 4)), 0),
            0)
        if meanFlag:
            return (cp.mean())
        else:
            return cp

    def clusterShadeGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        cs = numpy.sum(
            numpy.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 3)), 0),
            0)
        if meanFlag:
            return (cs.mean())
        else:
            return cs

    def clusterTendencyGLCM(self, P_glcm, sumMatrix, ux, uy, meanFlag=True):
        # Need to validate function
        ct = numpy.sum(
            numpy.sum((P_glcm * ((sumMatrix[:, :, None, None] - ux[None, None, :, :] - uy[None, None, :, :]) ** 2)), 0),
            0)
        if meanFlag:
            return (ct.mean())
        else:
            return ct

    def contrastGLCM(self, P_glcm, diffMatrix, meanFlag=True):
        cont = numpy.sum(numpy.sum((P_glcm * (diffMatrix[:, :, None, None] ** 2)), 0), 0)
        if meanFlag:
            return (cont.mean())
        else:
            return cont

    def correlationGLCM(self, P_glcm, prodMatrix, ux, uy, sigx, sigy, meanFlag=True):
        # Need to validate function
        uxy = ux * uy
        sigxy = sigx * sigy
        corr = numpy.sum(
            numpy.sum(((P_glcm * prodMatrix[:, :, None, None] - uxy[None, None, :, :]) / (sigxy[None, None, :, :])), 0),
            0)
        if meanFlag:
            return (corr.mean())
        else:
            return corr

    def differenceEntropyGLCM(self, pxSuby, eps, meanFlag=True):
        difent = numpy.sum((pxSuby * numpy.where(pxSuby != 0, numpy.log2(pxSuby), numpy.log2(eps))), 0)
        if meanFlag:
            return (difent.mean())
        else:
            return difent

    def dissimilarityGLCM(self, P_glcm, diffMatrix, meanFlag=True):
        dis = numpy.sum(numpy.sum((P_glcm * diffMatrix[:, :, None, None]), 0), 0)
        if meanFlag:
            return (dis.mean())
        else:
            return dis

    def energyGLCM(self, P_glcm, meanFlag=True):
        ene = numpy.sum(numpy.sum((P_glcm ** 2), 0), 0)
        if meanFlag:
            return (ene.mean())
        else:
            return ene

    def entropyGLCM(self, P_glcm, pxy, eps, meanFlag=True):
        ent = -1 * numpy.sum(numpy.sum((P_glcm * numpy.where(pxy != 0, numpy.log2(pxy), numpy.log2(eps))), 0), 0)
        if meanFlag:
            return (ent.mean())
        else:
            return ent

    def homogeneity1GLCM(self, P_glcm, diffMatrix, meanFlag=True):
        homo1 = numpy.sum(numpy.sum((P_glcm / (1 + diffMatrix[:, :, None, None])), 0), 0)
        if meanFlag:
            return (homo1.mean())
        else:
            return homo1

    def homogeneity2GLCM(self, P_glcm, diffMatrix, meanFlag=True):
        homo2 = numpy.sum(numpy.sum((P_glcm / (1 + diffMatrix[:, :, None, None] ** 2)), 0), 0)
        if meanFlag:
            return (homo2.mean())
        else:
            return homo2

    def imc1GLCM(self, HXY, HXY1, HX, HY, meanFlag=True):
        imc1 = (self.HXY - self.HXY1) / numpy.max(([self.HX, self.HY]), 0)
        if meanFlag:
            return (imc1.mean())
        else:
            return imc1

            # def imc2GLCM(self,):
            # imc2[g,a] = ( 1-numpy.e**(-2*(HXY2[g,a]-HXY[g,a])) )**(0.5) #nan value too high

            # produces Nan(square root of a negative)
            # exponent = decimal.Decimal( -2*(HXY2[g,a]-self.HXY[g,a]) )
            # imc2.append( ( decimal.Decimal(1)-decimal.Decimal(numpy.e)**(exponent) )**(decimal.Decimal(0.5)) )

            # if meanFlag:
            # return (homo2.mean())
            # else:
            # return homo2

    def idmnGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        idmn = numpy.sum(numpy.sum((P_glcm / (1 + ((diffMatrix[:, :, None, None] ** 2) / (Ng ** 2)))), 0), 0)
        if meanFlag:
            return (idmn.mean())
        else:
            return idmn

    def idnGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        idn = numpy.sum(numpy.sum((P_glcm / (1 + (diffMatrix[:, :, None, None] / Ng))), 0), 0)
        if meanFlag:
            return (idn.mean())
        else:
            return idn

    def inverseVarianceGLCM(self, P_glcm, diffMatrix, Ng, meanFlag=True):
        maskDiags = numpy.ones(diffMatrix.shape, dtype=bool)
        maskDiags[numpy.diag_indices(Ng)] = False
        inv = numpy.sum((P_glcm[maskDiags] / (diffMatrix[:, :, None, None] ** 2)[maskDiags]), 0)
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
        sumavg = numpy.sum((kValuesSum[:, None, None] * pxAddy), 0)
        if meanFlag:
            return (sumavg.mean())
        else:
            return sumavg

    def sumEntropyGLCM(self, pxAddy, eps, meanFlag=True):
        sumentr = (-1) * numpy.sum((pxAddy * numpy.where(pxAddy != 0, numpy.log2(pxAddy), numpy.log2(eps))), 0)
        if meanFlag:
            return (sumentr.mean())
        else:
            return sumentr

    def sumVarianceGLCM(self, pxAddy, kValuesSum, meanFlag=True):
        sumvar = numpy.sum((pxAddy * ((kValuesSum[:, None, None] - kValuesSum[:, None, None] * pxAddy) ** 2)), 0)
        if meanFlag:
            return (sumvar.mean())
        else:
            return sumvar

    def varianceGLCM(self, P_glcm, ivector, u, meanFlag=True):
        vari = numpy.sum(numpy.sum((P_glcm * ((ivector[:, None] - u) ** 2)[:, None, None, :]), 0), 0)
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

        angles = numpy.array([(1, 0, 0),
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
            for angles_idx in xrange(directions):
                angle = angles[angles_idx]

                for distances_idx in xrange(distances.size):
                    distance = distances[distances_idx]

                    i = matrix[h, c, r]
                    i_idx = numpy.nonzero(grayLevels == i)

                    row = r + angle[2]
                    col = c + angle[1]
                    height = h + angle[0]

                    # Can introduce Parameter Option for reference voxel(i) and neighbor voxel(j):
                    # Intratumor only: i and j both must be in tumor ROI
                    # Tumor+Surrounding: i must be in tumor ROI but J does not have to be
                    if row >= 0 and row < rows and col >= 0 and col < cols:
                        if tuple((height, col, row)) in indices:
                            j = matrix[height, col, row]
                            j_idx = numpy.nonzero(grayLevels == j)
                            # if i >= grayLevels.min and i <= grayLevels.max and j >= grayLevels.min and j <= grayLevels.max:
                            out[i_idx, j_idx, distances_idx, angles_idx] += 1
            # Check if the user has cancelled the process
            if iteration % 10 == 0:
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
        self.Nr = numpy.max(self.parameterMatrix.shape)
        self.Np = self.parameterValues.size
        self.eps = numpy.spacing(1)

        self.P_glrl = numpy.zeros(
            (self.Ng, self.Nr, self.angles))  # maximum run length in P matrix initialized to highest gray level
        self.P_glrl = self.calculate_glrl(self.grayLevels, self.Ng, self.parameterMatrix,
                                          self.parameterMatrixCoordinates, self.angles, self.P_glrl)

        self.sumP_glrl = numpy.sum(numpy.sum(self.P_glrl, 0), 0) + self.eps
        self.ivector = numpy.arange(self.Ng) + 1
        self.jvector = numpy.arange(self.Nr) + 1

    def shortRunEmphasis(self, P_glrl, jvector, sumP_glrl, meanFlag=True):
        try:
            sre = numpy.sum(numpy.sum((P_glrl / ((jvector ** 2)[None, :, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            sre = 0
        if meanFlag:
            return (sre.mean())
        else:
            return sre

    def longRunEmphasis(self, P_glrl, jvector, sumP_glrl, meanFlag=True):
        try:
            lre = numpy.sum(numpy.sum((P_glrl * ((jvector ** 2)[None, :, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lre = 0
        if meanFlag:
            return (lre.mean())
        else:
            return lre

    def grayLevelNonUniformity(self, P_glrl, sumP_glrl, meanFlag=True):
        try:
            gln = numpy.sum((numpy.sum(P_glrl, 1) ** 2), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            gln = 0
        if meanFlag:
            return (gln.mean())
        else:
            return gln

    def runLengthNonUniformity(self, P_glrl, sumP_glrl, meanFlag=True):
        try:
            rln = numpy.sum((numpy.sum(P_glrl, 0) ** 2), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            rln = 0
        if meanFlag:
            return (rln.mean())
        else:
            return rln

    def runPercentage(self, P_glrl, Np, meanFlag=True):
        try:
            rp = numpy.sum(numpy.sum((P_glrl / (Np)), 0), 0)
        except ZeroDivisionError:
            rp = 0
        if meanFlag:
            return (rp.mean())
        else:
            return rp

    def lowGrayLevelRunEmphasis(self, P_glrl, ivector, sumP_glrl, meanFlag=True):
        try:
            lglre = numpy.sum(numpy.sum((P_glrl / ((ivector ** 2)[:, None, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lglre = 0
        if meanFlag:
            return (lglre.mean())
        else:
            return lglre

    def highGrayLevelRunEmphasis(self, P_glrl, ivector, sumP_glrl, meanFlag=True):
        try:
            hglre = numpy.sum(numpy.sum((P_glrl * ((ivector ** 2)[:, None, None])), 0), 0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            hglre = 0
        if meanFlag:
            return (hglre.mean())
        else:
            return hglre

    def shortRunLowGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            srlgle = numpy.sum(numpy.sum((P_glrl / ((jvector ** 2)[None, :, None] * (ivector ** 2)[:, None, None])), 0),
                               0) / (sumP_glrl[None, None, :])
        except ZeroDivisionError:
            srlgle = 0
        if meanFlag:
            return (srlgle.mean())
        else:
            return srlgle

    def shortRunHighGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            srhgle = numpy.sum(
                numpy.sum(((P_glrl * (ivector ** 2)[:, None, None]) / ((jvector ** 2)[None, :, None])), 0), 0) / (
                         sumP_glrl[None, None, :])
        except ZeroDivisionError:
            srhgle = 0
        if meanFlag:
            return (srhgle.mean())
        else:
            return srhgle

    def longRunLowGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            lrlgle = numpy.sum(
                numpy.sum(((P_glrl * (jvector ** 2)[None, :, None]) / ((ivector ** 2)[:, None, None])), 0), 0) / (
                         sumP_glrl[None, None, :])
        except ZeroDivisionError:
            lrlgle = 0
        if meanFlag:
            return (lrlgle.mean())
        else:
            return lrlgle

    def longRunHighGrayLevelEmphasis(self, P_glrl, ivector, jvector, sumP_glrl, meanFlag=True):
        try:
            lrhgle = numpy.sum(numpy.sum((P_glrl * (ivector ** 2)[:, None, None] * (jvector ** 2)[None, :, None]), 0),
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
        # Use filter(lambda x: numpy.nonzero(x)[0].size>1, ....) to filter 1D arrays of size < 2 or value == 0 or padValue

        # Should change from nonzero() to filter for the padValue specifically (NaN, eps, etc)

        # (1,0,0), #(-1,0,0),
        aDiags = reduce(lambda x, y: x + y, [a.tolist() for a in numpy.transpose(matrix, (1, 2, 0))])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, aDiags))

        # (0,1,0), #(0,-1,0),
        bDiags = reduce(lambda x, y: x + y, [a.tolist() for a in numpy.transpose(matrix, (0, 2, 1))])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, bDiags))

        # (0,0,1), #(0,0,-1),
        cDiags = reduce(lambda x, y: x + y, [a.tolist() for a in numpy.transpose(matrix, (0, 1, 2))])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, cDiags))

        # (1,1,0),#(-1,-1,0),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        dDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 1).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, dDiags))

        # (1,0,1), #(-1,0-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[2]

        eDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 2).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, eDiags))

        # (0,1,1), #(0,-1,-1),
        lowBound = -matrix.shape[1] + 1
        highBound = matrix.shape[2]

        fDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 1, 2).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, fDiags))

        # (1,-1,0), #(-1,1,0),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        gDiags = reduce(lambda x, y: x + y,
                        [matrix[:, ::-1, :].diagonal(a, 0, 1).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, gDiags))

        # (-1,0,1), #(1,0,-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[2]

        hDiags = reduce(lambda x, y: x + y,
                        [matrix[:, :, ::-1].diagonal(a, 0, 2).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, hDiags))

        # (0,1,-1), #(0,-1,1),
        lowBound = -matrix.shape[1] + 1
        highBound = matrix.shape[2]

        iDiags = reduce(lambda x, y: x + y,
                        [matrix[:, :, ::-1].diagonal(a, 1, 2).tolist() for a in xrange(lowBound, highBound)])
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, iDiags))

        # (1,1,1), #(-1,-1,-1)
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        jDiags = [numpy.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix.diagonal(a, 0, 1) for a in xrange(lowBound, highBound)] for x in
                  xrange(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, jDiags))

        # (-1,1,-1), #(1,-1,1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        kDiags = [numpy.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, ::-1, :].diagonal(a, 0, 1) for a in xrange(lowBound, highBound)] for x in
                  xrange(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, kDiags))

        # (1,1,-1), #(-1,-1,1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        lDiags = [numpy.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, :, ::-1].diagonal(a, 0, 1) for a in xrange(lowBound, highBound)] for x in
                  xrange(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, lDiags))

        # (-1,1,1), #(1,-1,-1),
        lowBound = -matrix.shape[0] + 1
        highBound = matrix.shape[1]

        mDiags = [numpy.diagonal(h, x, 0, 1).tolist() for h in
                  [matrix[:, ::-1, ::-1].diagonal(a, 0, 1) for a in xrange(lowBound, highBound)] for x in
                  xrange(-h.shape[0] + 1, h.shape[1])]
        matrixDiagonals.append(filter(lambda x: numpy.nonzero(x)[0].size > 1, mDiags))

        # [n for n in mDiags if numpy.nonzero(n)[0].size>1] instead of filter(lambda x: numpy.nonzero(x)[0].size>1, mDiags)?

        # Run-Length Encoding (rle) for the 13 list of diagonals (1 list per 3D direction/angle)
        for angle in xrange(0, len(matrixDiagonals)):
            P = P_out[:, :, angle]
            for diagonal in matrixDiagonals[angle]:
                diagonal = numpy.array(diagonal, dtype='int')
                pos, = numpy.where(
                    numpy.diff(diagonal) != 0)  # can use instead of using map operator._ on np.where tuples
                pos = numpy.concatenate(([0], pos + 1, [len(diagonal)]))

                # a or pos[:-1] = run start #b or pos[1:] = run stop #diagonal[a] is matrix value
                # adjust condition for pos[:-1] != padVal = 0 to != padVal = eps or NaN or whatever pad value
                rle = zip([n for n in diagonal[pos[:-1]] if n != padVal], pos[1:] - pos[:-1])
                rle = [[numpy.where(grayLevels == x)[0][0], y - 1] for x, y in
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