from cip_python.phenotypes import nodule_features
# from ..phenotypes import nodule_features
import os
import collections
import numpy as np
import time
import SimpleITK as sitk
from argparse import ArgumentParser
import subprocess
import operator
import math
import vtk
import csv
import itertools

class NoduleSegmenter:

    def __init__(self, input_ct, nodule_lm, seed_point, l_type, feature_classes, csv_file_path):
        self.WORKING_MODE_HUMAN = 0
        self.WORKING_MODE_SMALL_ANIMAL = 1
        self.MAX_TUMOR_RADIUS = 30

        self._input_ct = input_ct
        self._nodule_lm = nodule_lm
        self._seed_point = seed_point
        self._lesion_type = l_type
        self._feature_classes = feature_classes
        self._current_distance_maps = {}
        self._analysis_results = dict()
        self._csv_file_path = csv_file_path
        self._analyzedSpheres = set()

    def noduleAnalysis(self, cid, sphere_radius, whole_lm=None):
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
            self._analysis_results[keyName] = collections.OrderedDict()
            self._analysis_results[keyName] = self.runAnalysis(whole_lm, nodule_lm_array, self._analysis_results[keyName],
                                                               selectedMainFeaturesKeys, selectedFeatureKeys)

            # Print analysis results
            print(self._analysis_results[keyName])

            if sphere_radius is not None:
                for sr in sphere_radius:
                    if "Parenchymal Volume" in selectedMainFeaturesKeys:
                        # If the parenchymal volume analysis is required, we need the numpy array representing the whole
                        # emphysema segmentation labelmap
                        labelmapWholeVolumeArray = sitk.GetArrayFromImage(whole_lm)
                    else:
                        labelmapWholeVolumeArray = None

                    self._current_distance_maps[cid] = self.getCurrentDistanceMap()
                    self.runAnalysisSphere(cid, self._current_distance_maps[cid], sr, selectedMainFeaturesKeys,
                                           selectedFeatureKeys, labelmapWholeVolumeArray)
                    self._analyzedSpheres.add(sr)

        finally:
            # print 'Finished'
            self.saveReport(cid)#, self._seed_point, self._lesion_type, self._analysis_results)

    def runAnalysis(self, whole_lm, n_lm_array, results_storage, feature_categories_keys, feature_keys):
        t1 = time.time()
        i_ct_array = sitk.GetArrayFromImage(self._input_ct)
        targetVoxels, targetVoxelsCoordinates = self.tumorVoxelsAndCoordinates(n_lm_array, i_ct_array)
        print("Time to calculate tumorVoxelsAndCoordinates: {0} seconds".format(time.time() - t1))
        print np.shape(targetVoxels)
        print np.shape(targetVoxelsCoordinates)
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
            firstOrderStatistics = nodule_features.FirstOrderStatistics(targetVoxels, bins, numGrayLevels, feature_keys)
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
            morphologyStatistics = nodule_features.MorphologyStatistics(self._input_ct.GetSpacing(), matrixSA,
                                                                        matrixSACoordinates, targetVoxels, feature_keys)
            results = morphologyStatistics.EvaluateFeatures()
            results_storage.update(results)

        # Texture Features(GLCM)
        if "Texture: GLCM" in feature_categories_keys:
            textureFeaturesGLCM = nodule_features.TextureGLCM(grayLevels, numGrayLevels, matrix, matrixCoordinates,
                                                              targetVoxels, feature_keys)
            results = textureFeaturesGLCM.EvaluateFeatures()
            results_storage.update(results)

        # Texture Features(GLRL)
        if "Texture: GLRL" in feature_categories_keys:
            textureFeaturesGLRL = nodule_features.TextureGLRL(grayLevels, numGrayLevels, matrix, matrixCoordinates,
                                                              targetVoxels, feature_keys)
            results = textureFeaturesGLRL.EvaluateFeatures()
            results_storage.update(results)

        # Geometrical Measures
        if "Geometrical Measures" in feature_categories_keys:
            geometricalMeasures = nodule_features.GeometricalMeasures(self._input_ct.GetSpacing(), matrix,
                                                                      matrixCoordinates, targetVoxels, feature_keys)
            results = geometricalMeasures.EvaluateFeatures()
            results_storage.update(results)

        # Renyi Dimensions
        if "Renyi Dimensions" in feature_categories_keys:
            # extend padding to dimension lengths equal to next power of 2
            maxDims = tuple([int(pow(2, math.ceil(np.log2(np.max(matrix.shape)))))] * 3)
            matrixPadded, matrixPaddedCoordinates = self.padMatrix(matrix, matrixCoordinates, maxDims, targetVoxels)
            renyiDimensions = nodule_features.RenyiDimensions(matrixPadded, matrixPaddedCoordinates, feature_keys)
            results = renyiDimensions.EvaluateFeatures()
            results_storage.update(results)

        # Parenchymal Volume
        if "Parenchymal Volume" in feature_categories_keys:
            parenchyma_lm_array = sitk.GetArrayFromImage(whole_lm)
            parenchymalVolume = nodule_features.ParenchymalVolume(parenchyma_lm_array, n_lm_array,
                                                                  input_ct.GetSpacing(), feature_keys)
            results = parenchymalVolume.EvaluateFeatures()
            results_storage.update(results)

        # filter for user-queried features only
        results_storage = collections.OrderedDict((k, results_storage[k]) for k in feature_keys)

        return results_storage

    def runAnalysisSphere(self, cid, dist_map, radius, selectedMainFeaturesKeys, selectedFeatureKeys,
                          parenchymaWholeVolumeArray=None):
        """ Run the selected features for an sphere of radius r (excluding the nodule itself)
        @param cid: case_id
        @param dist_map: distance map
        @param radius:
        @param parenchymaWholeVolumeArray: parenchyma volume (only used in parenchyma analysis). Numpy array
        """
        keyName = "{0}_r{1}".format(cid, int(radius))
        sphere_lm_array = self.getSphereLabelMapArray(dist_map, radius)
        if sphere_lm_array.max() == 0:
            # Nothing to analyze
            results = {}
            for key in selectedFeatureKeys:
                results[key] = 0
            self._analysis_results[keyName] = results
        else:
            self._analysis_results[keyName] = collections.OrderedDict()
            self.runAnalysis(parenchymaWholeVolumeArray, sphere_lm_array, self._analysis_results[keyName],
                             selectedMainFeaturesKeys, selectedFeatureKeys)

            print("********* Results for the sphere of radius {0}:".format(radius))
            print(self._analysis_results[keyName])

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
        # numpy version 1.7 has np.pad function

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
        :param np_array: numpy array
        :param labelId: label id (default = 1)
        :return: numpy array with the coordinates (int format)
        """
        mean = np.mean(np.where(np_array == labelId), axis=1)
        return np.asarray(np.round(mean, 0), np.int)

    def vtk_numpy_coordinate(self, vtk_coordinate):
        """ Adapt a coordinate in VTK to a numpy array format handled by VTK (ie in a reversed order)
        :param itk_coordinate: coordinate in VTK (xyz)
        :return: coordinate in numpy (zyx)
        """
        l = list(vtk_coordinate)
        l.reverse()
        return l

    def numpy_itk_coordinate(self, numpy_coordinate, convert_to_int=True):
        """ Adapt a coordinate in numpy to a ITK format (ie in a reversed order and converted to int type)
        :param numpy_coordinate: coordinate in numpy (zyx)
        :param convert_to_int: convert the coordinate to int type, needed for SimpleITK image coordinates
        :return: coordinate in ITK (xyz)
        """
        if convert_to_int:
            return [int(numpy_coordinate[2]), int(numpy_coordinate[1]), int(numpy_coordinate[0])]
        return [numpy_coordinate[2], numpy_coordinate[1], numpy_coordinate[0]]

    def getCurrentDistanceMap(self):
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
        seeds = [self.numpy_itk_coordinate(centroid)]
        fastMarchingFilter.SetTrialPoints(seeds)
        output = fastMarchingFilter.Execute(sitkImage)
        # self._current_distance_maps[cid] = sitk.GetArrayFromImage(output)

        return sitk.GetArrayFromImage(output)

    def getSphereLabelMapArray(self, dm, radius):
        """ Get a labelmap numpy array that contains a sphere centered in the nodule centroid, with radius "radius" and that
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
        sphere_lm_array[dm <= radius] = 1
        # Exclude the nodule
        sphere_lm_array[nodule_lm_array == 1] = 0
        return sphere_lm_array

    def saveReport(self, cid): #, seed, type, analysisResults):
        """ Save the current values in a persistent csv file
        """
        keyName = cid
        self.saveBasicData(keyName)
        self.saveCurrentValues(self._analysis_results[keyName])

        # Get all the spheres for this nodule
        for rad in self._analyzedSpheres:
            keyName = "{}_r{}".format(cid, int(rad))
            self.saveBasicData(keyName)
            self.saveCurrentValues(self._analysis_results[keyName])

    def saveBasicData(self, keyName):
        date = time.strftime("%Y/%m/%d %H:%M:%S")
        # noduleKeys = self.logic.getAllNoduleKeys(self.currentVolume)
        # for noduleIndex in noduleKeys:
        self._analysis_results[keyName]["CaseId"] = keyName
        self._analysis_results[keyName]["Date"] = date
        # d[keyName]["Nodule"] = noduleIndex
        # d[keyName]["Threshold"] = self.logic.marchingCubesFilters[(volume.GetID(), noduleIndex)].GetValue(0) \
        #     if (volume.GetID(), noduleIndex) in self.logic.marchingCubesFilters else str(self.logic.defaultThreshold)
        self._analysis_results[keyName]["LesionType"] = self._lesion_type
        self._analysis_results[keyName]["Seeds_LPS"] = self._seed_point

    def saveCurrentValues(self, analysis_results):
        """ Save a new row of information in the current csv file that stores the data  (from a dictionary of items)
        :param kwargs: dictionary of values
        """
        # Check that we have all the "columns"
        storedColumnNames = ["CaseId", "Date", "LesionType", "Seeds_LPS"]
        # Create a single features list with all the "child" features
        storedColumnNames.extend(itertools.chain.from_iterable(self._feature_classes.itervalues()))

        # for key in results:
        #     if key not in columnNames:
        #         print("WARNING: Column {0} is not included in the list of columns".format(key))
        # Add the values in the right order (there are not obligatory fields)
        orderedColumns = []
        # Always add a timestamp as the first value
        orderedColumns.append(time.strftime("%Y/%m/%d %H:%M:%S"))
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

def runNoduleSegmentation(i_ct, i_ct_filename, max_rad, seed, o_lm):
    """ Run the nodule segmentation through a CLI
    """
    tmpCommand = "GenerateLesionSegmentation -i %(in)s -o %(out)s --seeds %(sd)s --maximumRadius %(maxrad)f -f"
    tmpCommand = tmpCommand % {'in': i_ct_filename, 'out': o_lm, 'sd': seed, 'maxrad': max_rad}
    # tmpCommand = os.path.join(path['CIP_PATH'], tmpCommand)
    subprocess.call(tmpCommand, shell=True)

    nodule_segm_image = sitk.ReadImage(o_lm)
    nodule_segm_image = sitk.GetArrayFromImage(nodule_segm_image)
    nodule_segm_image[nodule_segm_image > 0] = 1
    nodule_segm_image[nodule_segm_image < 0] = 0
    sitkImage = sitk.GetImageFromArray(nodule_segm_image)
    sitkImage.SetSpacing(i_ct.GetSpacing())
    sitkImage.SetOrigin(i_ct.GetOrigin())
    sitk.WriteImage(sitkImage, o_lm)

def write_csv_first_row(csv_file_path):
    columnNames = ["Timestamp", "CaseId", "Date", "LesionType", "Seeds_LPS"]
    featureClasses = collections.OrderedDict()
    featureClasses["First-Order Statistics"] = ["Voxel Count", "Gray Levels", "Energy", "Entropy",
                                                  "Minimum Intensity", "Maximum Intensity",
                                                  "Mean Intensity",
                                                  "Median Intensity", "Range", "Mean Deviation",
                                                  "Root Mean Square", "Standard Deviation",
                                                  "Ventilation Heterogeneity",
                                                  "Skewness", "Kurtosis", "Variance", "Uniformity"]
    featureClasses["Morphology and Shape"] = ["Volume mm^3", "Volume cc", "Surface Area mm^2",
                                                       "Surface:Volume Ratio", "Compactness 1", "Compactness 2",
                                                       "Maximum 3D Diameter", "Spherical Disproportion",
                                                       "Sphericity"]
    featureClasses["Texture: GLCM"] = ["Autocorrelation", "Cluster Prominence", "Cluster Shade",
                                                "Cluster Tendency", "Contrast", "Correlation",
                                                "Difference Entropy",
                                                "Dissimilarity", "Energy (GLCM)", "Entropy(GLCM)",
                                                "Homogeneity 1",
                                                "Homogeneity 2", "IMC1", "IDMN", "IDN", "Inverse Variance",
                                                "Maximum Probability", "Sum Average", "Sum Entropy",
                                                "Sum Variance",
                                                "Variance (GLCM)"]  # IMC2 missing
    featureClasses["Texture: GLRL"] = ["SRE", "LRE", "GLN", "RLN", "RP", "LGLRE", "HGLRE", "SRLGLE",
                                                "SRHGLE", "LRLGLE", "LRHGLE"]
    featureClasses["Geometrical Measures"] = ["Extruded Surface Area", "Extruded Volume",
                                                       "Extruded Surface:Volume Ratio"]
    featureClasses["Renyi Dimensions"] = ["Box-Counting Dimension", "Information Dimension",
                                                   "Correlation Dimension"]

    featureClasses["Parenchymal Volume"] = nodule_features.ParenchymalVolume.getAllEmphysemaDescriptions()

    columnNames.extend(itertools.chain.from_iterable(featureClasses.itervalues()))
    with open(csv_file_path, 'a+b') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columnNames)

if __name__ == "__main__":
    desc = """This module allows to segment benign nodules and tumors in the lung.
            Besides, it analyzes a lot of different features inside the nodule and in its surroundings,
            in concentric spheres of different radius centered in the centroid of the nodule"""

    parser = ArgumentParser(description=desc)
    parser.add_argument('--in_ct',
                        help='Input CT file', dest='in_ct', metavar='<string>',
                        default=None)
    parser.add_argument('--seed',
                        help='Coordinates (x,y,z) of lesion location (RAS).', dest='seed_point',
                        metavar='<string>', default='(0.0,0.0,0.0)')
    parser.add_argument('--type',
                        help='Type for each lesion indicated. Choose between Unknown, \
                        Nodule and Tumor types',
                        dest='type', metavar='<string>', default='Unknown')
    parser.add_argument('--max_rad',
                        help='Maximum radius (mm) for the lesion. Recommended: 30 mm \
                        for humans and 3 mm for small animals',
                        dest='max_rad', metavar='<string>', default=30)
    parser.add_argument('--n_lm',
                        help='Nodule labelmap. If labelmap exists, it will be used for \
                        analysis. Otherwise, nodule will be segmented first.', dest='n_lm',
                        metavar='<string>', default=None)
    parser.add_argument('--out_csv',
                        help='CSV file to save nodule analysis.', dest='csv_file', metavar='<string>',
                        default=None)
    # parser.add_argument('--min_th',
    #                   help='Min threshold for nodule segmentation.',
    #                   dest='min_th', metavar='<string>', default=-50)
    # parser.add_argument('--max_th',
    #                   help='Max threshold for nodule segmentation.',
    #                   dest='max_th', metavar='<string>', default=50)
    parser.add_argument('--fos_feat',
                        help='First Order Statistics features. For computation of \
                        all fos features indicate all.',
                        dest='fos_features', metavar='<string>', default=None)
    parser.add_argument('--ms_feat',
                        help='Morphology and Shape features. For computation of \
                        all ms features indicate all.',
                        dest='ms_features', metavar='<string>', default=None)
    parser.add_argument('--glcm_feat',
                        help='Gray-Level Co-ocurrence Matrices features. For computation of \
                        all glcm features indicate all.',
                        dest='glcm_features', metavar='<string>', default=None)
    parser.add_argument('--glrl_feat',
                        help='Gray-Level Run Length features. For computation of \
                        all glrl features indicate all.',
                        dest='glrl_features', metavar='<string>', default=None)
    parser.add_argument('--renyi_dim',
                        help='Renyi Dimensions. For computation of all renyi dimensions indicate all.',
                        dest='renyi_dimensions', metavar='<string>', default=None)
    parser.add_argument('--geom_meas',
                        help='Geometrical Measures. For computation of all renyi dimensions indicate all.',
                        dest='geom_measures', metavar='<string>', default=None)
    parser.add_argument('--sphere_rad',
                        help='Radius(es) for Sphere computation.', metavar='<int>', dest='sphere_rad',
                        default=None)
    parser.add_argument('--par_an',
                        help='Select this flag to compute parenchymal volume analysis.',
                        dest='par_an', action='store_true')
    parser.add_argument('-par_lm',
                        help='Parenchyma labelmap for parenchymal volume analysis.',
                        dest='par_lm', metavar='<string>', default=None)
    parser.add_argument('-par_feat',
                        help='Parenchymal volume features. For computation of \
                              all features indicate all.',
                        dest='par_features', metavar='<string>', default=None)

    options = parser.parse_args()

    fileparts = os.path.splitext(options.in_ct)
    case_id = fileparts[0].split('/')[-1:][0]

    input_ct = sitk.ReadImage(options.in_ct)
    seed_point = [float(s) for s in options.seed_point.split(',')]
    seed_point = ras_to_lps(seed_point)
    seed_point = '{},{},{}'.format(seed_point[0],seed_point[1],seed_point[2])

    lesion_type = options.type
    max_radius = int(options.max_rad)

    if not os.path.exists(options.n_lm):
        runNoduleSegmentation(input_ct, options.in_ct, max_radius, seed_point, options.n_lm)
    nodule_lm = sitk.ReadImage(options.n_lm)

    featureClasses = collections.OrderedDict()
    if options.fos_features == 'all':
        featureClasses["First-Order Statistics"] = ["Voxel Count", "Gray Levels", "Energy", "Entropy",
                                                    "Minimum Intensity", "Maximum Intensity", "Mean Intensity",
                                                    "Median Intensity", "Range", "Mean Deviation", "Root Mean Square",
                                                    "Standard Deviation", "Ventilation Heterogeneity", "Skewness",
                                                    "Kurtosis", "Variance", "Uniformity"]
    elif options.fos_features is not None:
        featureClasses["First-Order Statistics"] = [ff for ff in str.split(options.fos_features, ',')]

    if options.ms_features == 'all':
        featureClasses["Morphology and Shape"] = ["Volume mm^3", "Volume cc", "Surface Area mm^2", "Surface:Volume Ratio",
                                                  "Compactness 1", "Compactness 2", "Maximum 3D Diameter",
                                                  "Spherical Disproportion", "Sphericity"]
    elif options.ms_features is not None:
        featureClasses["Morphology and Shape"] = [ff for ff in str.split(options.ms_features, ',')]

    if options.glcm_features == 'all':
        featureClasses["Texture: GLCM"] = ["Autocorrelation", "Cluster Prominence", "Cluster Shade", "Cluster Tendency",
                                           "Contrast", "Correlation", "Difference Entropy", "Dissimilarity",
                                           "Energy (GLCM)", "Entropy(GLCM)", "Homogeneity 1", "Homogeneity 2", "IMC1",
                                           "IDMN", "IDN", "Inverse Variance", "Maximum Probability", "Sum Average",
                                           "Sum Entropy", "Sum Variance", "Variance (GLCM)"]  # IMC2 missing
    elif options.glcm_features is not None:
        featureClasses["Texture: GLCM"] = [ff for ff in str.split(options.glcm_features, ',')]

    if options.glrl_features == 'all':
        featureClasses["Texture: GLRL"] = ["SRE", "LRE", "GLN", "RLN", "RP", "LGLRE", "HGLRE", "SRLGLE", "SRHGLE",
                                           "LRLGLE", "LRHGLE"]
    elif options.glrl_features is not None:
        featureClasses["Texture: GLRL"] = [ff for ff in str.split(options.glrl_features, ',')]

    if options.geom_measures == 'all':
        featureClasses["Geometrical Measures"] = ["Extruded Surface Area", "Extruded Volume",
                                                  "Extruded Surface:Volume Ratio"]
    elif options.geom_measures is not None:
        featureClasses["Geometrical Measures"] = [ff for ff in str.split(options.geom_measures, ',')]

    if options.renyi_dimensions == 'all':
        featureClasses["Renyi Dimensions"] = ["Box-Counting Dimension", "Information Dimension", "Correlation Dimension"]
    elif options.renyi_dimensions is not None:
        featureClasses["Renyi Dimensions"] = [ff for ff in str.split(options.renyi_dimensions, ',')]

    if options.sphere_rad is not None:
        sphere_rad = [int(r) for r in options.sphere_rad.split(',')]
    else:
        sphere_rad = None

    if not os.path.exists(options.csv_file):
        write_csv_first_row(options.csv_file)

    ns = NoduleSegmenter(input_ct, nodule_lm, seed_point, lesion_type, featureClasses, options.csv_file)

    if options.par_an:
        parenchyma_lm = sitk.ReadImage(options.par_lm)
        if options.par_features == 'all':
            featureClasses["Parenchymal Volume"] = nodule_features.ParenchymalVolume.getAllEmphysemaDescriptions()
        elif options.par_features is not None:
            featureClasses["Parenchymal Volume"] = [ff for ff in str.split(options.par_features, ',')]

        ns.noduleAnalysis(case_id, sphere_rad, whole_lm=parenchyma_lm)
    else:
        ns.noduleAnalysis(case_id, sphere_rad)


