import numpy as np

from sklearn.cluster import DBSCAN, MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

class ClusterParticles:

  def __init__(self,in_particles,out_particles_collection,method='DBSCAN'):
    assert method 
    self._in_vtk=in_particles
    self._out_vtk_collection=out_particles_collection
    self._number_of_clusters = -1
    self._method = 'DBSCAN'
    self._centroids=np.array([])
    self._unique_labels=np.array([])
  
  def execute(self):
    
    #Get points from vtk file as numpy array
    points = vtk_to_numpy(self._in_vtk.GetPoints().GetData())
    
    points=StandardScaler().fit_transform(points)
    #Clustering
    if self._method == 'DBSCAN':
      db=DBSCAN(eps=0.3, min_samples=10).fit(points)
      core_samples = db.core_sample_indices_
      labels = db.labels_
    elif self._method == 'KMeans':
      kmean= KMeans(init='k-means++',n_clusters=self._number_of_clusters,n_init=10).fit(points)
      core_samples = kmeans.cluster_centers_
      labels = kmeans.labels_
    elif self._method == 'MiniBatchKMeans':
      mbk =  MiniBatchKMeans(init='k-means++', n_clusters=self._number_of_clusters, batch_size=20,
                             n_init=10, max_no_improvement=10, verbose=0).fit(points)
      labels = mbk.labels_
      core_samples = mbk.cluster_centers_

    unique_labels=set(labels)
    self._centroids = core_samples
    self._unique_labels = unique_labels
    
    #Save data for each cluster as a vtkPolyData
    for k in unique_labels:
      ids = np.argwhere(labels == k).flatten()
      self._out_vtk_collection.AddItem(self.extract_particles(ids))

  def extract_particles(self,ids):
    data=vtk.vtkPolyData()
    points=vtk_to_numpy(self._in_vtk.GetPoints().GetData())

    s_points=vtk.vtkPoints()
    
    #s_points.SetData(numpy_to_vtk(points[ids,:]))
    #s_points.SetNumberOfPoints(s_points.GetData().GetNumberOfTuples())
    s_p = points[ids,:]
    s_points.SetNumberOfPoints(s_p.shape[0])
    for kk in xrange(s_p.shape[0]):
      s_points.SetPoint(kk,s_p[kk,0],s_p[kk,1],s_p[kk,2])
    
    data.SetPoints(s_points)
    #Transfer point data and field data
    for pd,out_pd in zip([self._in_vtk.GetPointData(),self._in_vtk.GetFieldData()],[data.GetPointData(),data.GetFieldData()]):
      pd=self._in_vtk.GetPointData()
      for k in xrange(pd.GetNumberOfArrays()):
        arr=vtk_to_numpy(pd.GetArray(pd.GetArrayName(k)))
        if len(arr.shape) == 1:
          s_vtk_arr=numpy_to_vtk(arr[ids],1)
        else:
          s_vtk_arr=numpy_to_vtk(arr[ids,:],1)
        
        s_vtk_arr.SetName(pd.GetArrayName(k))
        out_pd.AddArray(s_vtk_arr)
    return data
                      
    #Method to do the extraction using a vtk pipeline (experimental with seg fault)
    def extract_using_vtk(self,ids):
      node=vtk.vtkSelectionNode()
      sel = vtk.vtkSelection()
      node.GetProperties().Set(vtk.vtkSelectionNode.CONTENT_TYPE(),\
                               vtk.vtkSelectionNode.INDICES)
      node.GetProperties().Set(vtk.vtkSelectionNode.FIELD_TYPE(),\
                               vtk.vtkSelectionNode.POINT)
      
      #Create Id Array with point Ids for each cluster
      vtk_ids=numpy_to_vtkIdTypeArray(ids)
      node.SetSelectionList(vtk_ids)
      #sel_filter = vtk.vtkExtractSelectedPolyDataIds()
      sel_filter = vtk.vtkExtractSelection()
      sel_filter.SetInput(0,self._in_vtk)
      sel_filter.SetInput(1,sel)
      sel_filter.Update()
      return sel_filter.GetOutput()
                      
if __name__ == "__main__":
  desc = """Cluster particles points"""
  
  import argparse
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-i',help='Input particle file (vtk) to cluster',
                    dest='in_file',metavar='<string>',default=None)
  parser.add_argument('--op',help='Output prefix name',
                    dest='output_prefix',metavar='<string>',default=None)
  parser.add_argument('--os',help='Output suffix name',
                    dest='output_suffix',metavar='<string>',default='.vtk')
  parser.add_argument('-s',help='Split particles in left/right lung',
                      dest='split_flag',action="store_true",default=False)
  
  op =  parser.parse_args()
  
  reader=vtk.vtkPolyDataReader()
  reader.SetFileName(op.in_file)
  reader.Update()
  
  input=reader.GetOutput()
  
  output_collection=vtk.vtkCollection()
  cluster=ClusterParticles(input,output_collection)

  if op.split_flag == True:
    cluster._number_of_clusters=2
    cluster._method='MiniBatchKMeans'
  else:
    cluster._method='DBSCAN'

  cluster.execute()

  if op.split_flag == True:
    #Find out which cluster is left vs right
    if cluster._centroids[0,0] > cluster._centroids[1,0]:
      tags=['left','right']
    else:
      tags=['right','left']
    for k,tag in zip([0,1],tags):
      writer=vtk.vtkPolyDataWriter()
      writer.SetInput(output_collection.GetItemAsObject(k))
      writer.SetFileName(op.output_prefix + '_%s%s' % (tag,op.output_suffix))
      writer.Update()
  else: 
    for k in xrange(output_collection.GetNumberOfItems()):
      writer=vtk.vtkPolyDataWriter()
      writer.SetInput(output_collection.GetItemAsObject(k))
      writer.SetFileName(op.output_prefix + '_cluster%03d%s' % (k,op.output_suffix) )
      writer.Update()

