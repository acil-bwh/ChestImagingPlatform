import numpy as np

from sklearn.cluster import DBSCAN, MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.decomposition import PCA
#  from sklearn import metrics
#  from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

import matplotlib.pyplot as plt

# NOTE: backward compatibility Python 2
try:
    # Python 2
    xrange
except NameError:
    # Python 3
    xrange = range


class ClusterParticles:

    def __init__(self,
                 in_particles,
                 out_particles_collection,
                 method='DBSCAN',
                 feature_extractor=None):
        assert method
        self._in_vtk = in_particles
        self._out_vtk_collection = out_particles_collection
        self._number_of_clusters = -1
        self._method = 'DBSCAN'
        self._centroids = np.array([])
        self._unique_labels = np.array([])
        if feature_extractor == None:
            self._feature_extractor = self.feature_extractor
        else:
            self._feature_extractor = feature_extractor

    def execute(self):

        # Get points from vtk file as numpy array
        features = self._feature_extractor(self._in_vtk)
        features = StandardScaler().fit_transform(features)

        # Clustering
        if self._method == 'DBSCAN':
            db = DBSCAN(eps=0.3, min_samples=10).fit(features)
            core_samples = db.core_sample_indices_
            labels = db.labels_
        elif self._method == 'KMeans':
            kmeans = KMeans(init='k-means++',
                            n_clusters=self._number_of_clusters,
                            n_init=50).fit(features)
            core_samples = kmeans.cluster_centers_
            labels = kmeans.labels_

        elif self._method == 'MiniBatchKMeans':
            mbk = MiniBatchKMeans(init='k-means++',
                                  n_clusters=self._number_of_clusters,
                                  batch_size=20,
                                  n_init=20,
                                  max_no_improvement=10,
                                  verbose=0).fit(features)
            labels = mbk.labels_
            core_samples = mbk.cluster_centers_
        elif self._method == 'SpectralClustering':
            sc = SpectralClustering(
                n_clusters=self._number_of_clusters).fit(features)
            labels = sc.labels_
            core_samples = np.zeros(
                [self._number_of_clusters, features.shape[1]])
            for ii in self._number_of_clusters:
                core_samples[ii, :] = np.means(features[labels, :], axis=0)

        unique_labels = set(labels)
        self._labels = labels
        self._centroids = core_samples
        self._unique_labels = unique_labels

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.scatter(features[:, 0],
                   features[:, 1],
                   c=labels,
                   marker='.',
                   cmap=plt.cm.jet,
                   linewidth=0)
        ax.grid(True)
        fig.savefig('test.png')

        # Save data for each cluster as a vtkPolyData
        for k in unique_labels:
            ids = np.argwhere(labels == k).flatten()
            print(labels.shape[0])
            print(ids.shape[0])
            print(self._in_vtk.GetNumberOfPoints())
            self._out_vtk_collection.AddItem(self.extract_particles(ids))

    def feature_extractor(self, in_vtk):
        points = vtk_to_numpy(in_vtk.GetPoints().GetData())
        return points

    def extract_particles(self, ids):
        data = vtk.vtkPolyData()
        points = vtk_to_numpy(self._in_vtk.GetPoints().GetData())

        s_points = vtk.vtkPoints()
        cell_arr = vtk.vtkCellArray()

        # s_points.SetData(numpy_to_vtk(points[ids,:]))
        # s_points.SetNumberOfPoints(s_points.GetData().GetNumberOfTuples())
        s_p = points[ids, :]
        s_points.SetNumberOfPoints(s_p.shape[0])
        cell_arr.SetNumberOfCells(s_p.shape[0])
        for kk in xrange(s_p.shape[0]):
            s_points.SetPoint(kk, s_p[kk, 0], s_p[kk, 1], s_p[kk, 2])
            cell_arr.InsertNextCell(1)
            cell_arr.InsertCellPoint(kk)

        data.SetPoints(s_points)
        data.SetVerts(cell_arr)

        # Transfer point data and field data
        for pd, out_pd in zip(
            [self._in_vtk.GetPointData(),
             self._in_vtk.GetFieldData()],
            [data.GetPointData(), data.GetFieldData()]):
            for k in xrange(pd.GetNumberOfArrays()):
                arr = vtk_to_numpy(pd.GetArray(pd.GetArrayName(k)))
                if arr.shape[0] < len(ids):
                    # Transfer directly without masking with sids
                    s_vtk_arr = numpy_to_vtk(arr, 1)
                else:
                    if len(arr.shape) == 1:
                        s_vtk_arr = numpy_to_vtk(arr[ids], 1)
                    else:
                        s_vtk_arr = numpy_to_vtk(arr[ids, :], 1)

                s_vtk_arr.SetName(pd.GetArrayName(k))
                out_pd.AddArray(s_vtk_arr)

        return data

        # Method to do the extraction using a vtk pipeline (experimental with
        # seg fault).
        def extract_using_vtk(self, ids):
            node = vtk.vtkSelectionNode()
            sel = vtk.vtkSelection()
            node.GetProperties().Set(vtk.vtkSelectionNode.CONTENT_TYPE(),\
                                     vtk.vtkSelectionNode.INDICES)
            node.GetProperties().Set(vtk.vtkSelectionNode.FIELD_TYPE(),\
                                     vtk.vtkSelectionNode.POINT)

            # Create Id Array with point Ids for each cluster
            vtk_ids = numpy_to_vtkIdTypeArray(ids)
            node.SetSelectionList(vtk_ids)
            # sel_filter = vtk.vtkExtractSelectedPolyDataIds()
            sel_filter = vtk.vtkExtractSelection()
            sel_filter.SetInput(0, self._in_vtk)
            sel_filter.SetInput(1, sel)
            sel_filter.Update()
            return sel_filter.GetOutput()


class LobeParticleLabeling():

    def __init__(self, in_vtk):
        self._in_vtk = in_vtk
        self._out_vtk = dict()
        self._out_vtk['LUL'] = None
        self._out_vtk['LLL'] = None
        self._out_vtk['RUL'] = None
        self._out_vtk['RLL'] = None
        self._out_vtk['RML'] = None
        self.cluster_tags = list()

    def execute(self):

        chest_region = list()
        chest_type = list()

        output_collection = vtk.vtkCollection()

        left_right_splitter = LeftRightParticleLabeling(input)
        leftright_output = left_right_splitter.execute()

        # Right splitter
        print("Right splitter")
        cluster = ClusterParticles(
            leftright_output['right'],
            output_collection,
            feature_extractor=self.feature_extractor_right)
        cluster._number_of_clusters = 3
        # cluster._method='MiniBatchKMeans'
        cluster._method = 'KMeans'

        cluster.execute()

        print("Done right clustering")

        points = vtk_to_numpy(leftright_output['right'].GetPoints().GetData())

        p_centroids = np.zeros([3, 3])
        for ii, ll in enumerate(cluster._unique_labels):
            p_centroids[ii, :] = np.mean(points[cluster._labels == ll, :],
                                         axis=0)

        # Sort centroids and get values
        sortval = np.sort(p_centroids[:, 2])
        indices = np.array(range(0, cluster._number_of_clusters))
        region_labels = ['RLL', 'RML', 'RUL']
        region_values = [6, 5, 4]

        for ii, ll in enumerate(cluster._unique_labels):
            idx_region = indices[sortval == p_centroids[ii, 2]][0]
            self.cluster_tags.append(region_labels[idx_region])
            chest_region.append(region_values[idx_region])
            chest_type.append(3)

        # Left splitter
        print("Left splitter")
        print(leftright_output['left'].GetNumberOfPoints())
        cluster = ClusterParticles(
            leftright_output['left'],
            output_collection,
            feature_extractor=self.feature_extractor_left)
        cluster._number_of_clusters = 2
        # cluster._method='MiniBatchKMeans'
        cluster._method = 'KMeans'

        cluster.execute()

        print("Done left clustering")
        points = vtk_to_numpy(leftright_output['left'].GetPoints().GetData())

        p_centroids = np.zeros([2, 3])
        for ii, ll in enumerate(cluster._unique_labels):
            p_centroids[ii, :] = np.mean(points[cluster._labels == ll, :],
                                         axis=0)

        # Sort centroids and get values
        sortval = np.sort(p_centroids[:, 2])
        indices = np.array(range(0, cluster._number_of_clusters))
        region_labels = ['LLL', 'LUL']
        region_values = [8, 7]

        for ii, ll in enumerate(cluster._unique_labels):
            idx_region = indices[sortval == p_centroids[ii, 2]][0]
            self.cluster_tags.append(region_labels[idx_region])
            chest_region.append(region_values[idx_region])
            chest_type.append(3)

        append = vtk.vtkAppendPolyData()
        for k, tag, cr, ct in zip(range(0, len(self.cluster_tags)),
                                  self.cluster_tags, chest_region, chest_type):
            print(k)
            self._out_vtk[tag] = output_collection.GetItemAsObject(k)
            chest_region_arr = vtk.vtkUnsignedCharArray()
            chest_region_arr.SetName('ChestRegion')
            chest_type_arr = vtk.vtkUnsignedCharArray()
            chest_type_arr.SetName('ChestType')
            n_p = self._out_vtk[tag].GetNumberOfPoints()
            chest_region_arr.SetNumberOfTuples(n_p)
            chest_type_arr.SetNumberOfTuples(n_p)
            for ii in xrange(self._out_vtk[tag].GetNumberOfPoints()):
                chest_region_arr.SetValue(ii, cr)
                chest_type_arr.SetValue(ii, ct)
            self._out_vtk[tag].GetPointData().AddArray(chest_region_arr)
            self._out_vtk[tag].GetPointData().AddArray(chest_type_arr)

            append.AddInputData(self._out_vtk[tag])

        append.Update()
        self._out_vtk['all'] = append.GetOutput()
        return self._out_vtk

    def feature_extractor_left(self, in_vtk):
        points = vtk_to_numpy(in_vtk.GetPoints().GetData())
        #  vec = vtk_to_numpy(in_vtk.GetPointData().GetArray("hevec0"))
        # features =np.concatenate((points, vec[:,0:1]),axis=1)
        features = points
        # features=StandardScaler().fit_transform(features)
        pca = PCA(n_components=3)
        pca.fit(features)
        features_t = pca.transform(features)
        return features_t[:, [0, 1]]
        # return features[:,[0,2]]

    def feature_extractor_right(self, in_vtk):
        points = vtk_to_numpy(in_vtk.GetPoints().GetData())
        vec = vtk_to_numpy(in_vtk.GetPointData().GetArray("hevec0"))
        features = np.concatenate((points, vec[:, [0, 1, 2]]), axis=1)
        # features = points
        # features=StandardScaler().fit_transform(features)
        pca = PCA(n_components=3)
        pca.fit(features)
        features_t = pca.transform(features)
        return features_t
        # return features[:,[0,2]]


class LeftRightParticleLabeling():

    def __init__(self, in_vtk):
        self._in_vtk = in_vtk
        self._out_vtk = dict()
        self._out_vtk['left'] = None
        self._out_vtk['right'] = None
        self._out_vtk['both'] = None
        self.cluster_tags = list()

    def execute(self):
        output_collection = vtk.vtkCollection()
        cluster = ClusterParticles(input,
                                   output_collection,
                                   feature_extractor=self.feature_extractor)
        cluster._number_of_clusters = 2
        # cluster._method='MiniBatchKMeans'
        cluster._method = 'KMeans'

        cluster.execute()

        points = vtk_to_numpy(input.GetPoints().GetData())

        p_centroids = np.zeros([2, 3])
        for ii, ll in enumerate(cluster._unique_labels):
            p_centroids[ii, :] = np.mean(points[cluster._labels == ll, :],
                                         axis=0)

        if p_centroids[0, 0] > p_centroids[1, 0]:
            self.cluster_tags = ['left', 'right']
            chest_region = [3, 2]
            chest_type = [3, 3]
        else:
            self.cluster_tags = ['right', 'left']
            chest_region = [2, 3]
            chest_type = [3, 3]

        append = vtk.vtkAppendPolyData()
        for k, tag, cr, ct in zip([0, 1], self.cluster_tags, chest_region,
                                  chest_type):
            self._out_vtk[tag] = output_collection.GetItemAsObject(k)
            chest_region_arr = vtk.vtkUnsignedCharArray()
            chest_region_arr.SetName('ChestRegion')
            chest_type_arr = vtk.vtkUnsignedCharArray()
            chest_type_arr.SetName('ChestType')
            n_p = self._out_vtk[tag].GetNumberOfPoints()
            chest_region_arr.SetNumberOfTuples(n_p)
            chest_type_arr.SetNumberOfTuples(n_p)
            for ii in xrange(self._out_vtk[tag].GetNumberOfPoints()):
                chest_region_arr.SetValue(ii, cr)
                chest_type_arr.SetValue(ii, ct)
            self._out_vtk[tag].GetPointData().AddArray(chest_region_arr)
            self._out_vtk[tag].GetPointData().AddArray(chest_type_arr)

            append.AddInputData(self._out_vtk[tag])

        append.Update()
        self._out_vtk['all'] = append.GetOutput()
        return self._out_vtk

    def feature_extractor(self, in_vtk):
        points = vtk_to_numpy(in_vtk.GetPoints().GetData())
        #  vec = vtk_to_numpy(in_vtk.GetPointData().GetArray("hevec0"))
        # features =np.concatenate((points, vec[:,0:1]),axis=1)
        features = points
        # features=StandardScaler().fit_transform(features)
        pca = PCA(n_components=3)
        pca.fit(features)
        features_t = pca.transform(features)
        return features_t[:, [0, 1]]


# return features[:,[0,2]]

if __name__ == "__main__":
    desc = """Cluster particles points"""

    import argparse
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i',
                        help='Input particle file (vtk) to cluster',
                        dest='in_file',
                        metavar='<string>',
                        default=None)
    parser.add_argument('--op',
                        help='Output prefix name',
                        dest='output_prefix',
                        metavar='<string>',
                        default=None)
    parser.add_argument('--os',
                        help='Output suffix name',
                        dest='output_suffix',
                        metavar='<string>',
                        default='.vtk')
    parser.add_argument('-s',
                        help='Split particles in left/right lung',
                        dest='split_flag',
                        action="store_true",
                        default=False)
    parser.add_argument('-l',
                        help='Split particles in lobes',
                        dest='lobe_flag',
                        action="store_true",
                        default=False)
    parser.add_argument(
        '--pd',
        help='save particles in a single vtk with region/type PointData array',
        dest='label_flag',
        default=False,
        action="store_true")

    op = parser.parse_args()

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(op.in_file)
    reader.Update()

    input = reader.GetOutput()

    if op.split_flag or op.lobe_flag == True:
        if op.split_flag == True:
            labeler = LeftRightParticleLabeling(input)
        else:
            labeler = LobeParticleLabeling(input)

        output = labeler.execute()
        if op.label_flag == True:
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(output['all'])
            writer.SetFileTypeToBinary()
            writer.SetFileName(op.output_prefix + op.output_suffix)
            writer.Update()
        else:
            for tag in labeler.cluster_tags:
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputData(output[tag])
                writer.SetFileName(op.output_prefix + '_%s%s' %
                                   (tag, op.output_suffix))
                writer.SetFileTypeToBinary()
                writer.Update()
    else:
        output_collection = vtk.vtkCollection()
        cluster = ClusterParticles(input, output_collection)
        cluster._method = 'DBSCAN'
        cluster.execute()
        for k in xrange(output_collection.GetNumberOfItems()):
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(output_collection.GetItemAsObject(k))
            writer.SetFileName(op.output_prefix + '_cluster%03d%s' %
                               (k, op.output_suffix))
            writer.SetFileTypeToBinary()
            writer.Update()
