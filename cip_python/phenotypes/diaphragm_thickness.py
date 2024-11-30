import vtk, math
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.stats import scoreatpercentile

from optparse import OptionParser
import os.path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DiaphragmThickness:

    def __init__(self, input_file, output_prefix,spacing,plot=True):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.spacing = spacing
        self.plot = plot
        self.sigma = 0.18

    def execute(self):
      readerD=vtk.vtkPolyDataReader()
      readerD.SetFileName(self.input_file)
      readerD.Update()
      dia=readerD.GetOutput()

      array_d =dict();
      for ff in ("scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2"):
        tmp=dia.GetFieldData().GetArray(ff)
        if isinstance(tmp,vtk.vtkDataArray) == False:
          tmp=dia.GetPointData().GetArray(ff)
        array_d[ff]=tmp
   
      scale_arr = vtk_to_numpy(array_d['scale'])
      d_radius = self.particle_radius_from_scale(scale_arr)
      print (d_radius)
      qq=[]
      qqname=[]
      qqname.append('CID')
      qq.append(os.path.split(self.output_prefix)[1])

      qq.append(np.mean(d_radius))
      qqname.append('leftLung DT mean')
      qq.append(np.std(d_radius))
      qqname.append('leftLung DT std')
      qq.append(np.min(d_radius))
      qqname.append('leftLung DT min')
      qq.append(np.max(d_radius))
      qqname.append('leftLung DT max')
      for percentile in (5,10,25,50,75,90,95):
        qq.append(scoreatpercentile(d_radius,percentile))
        qqname.append('leftLung DT perc'+str(percentile))
  
      df=pd.DataFrame(np.array(qq).reshape([1,len(qq)]),columns=qqname)
      df.to_csv(self.output_prefix+'_diaphragmPhenotypes.csv')
  
      if self.plot == True:
        xyz_arr = vtk_to_numpy(dia.GetPoints().GetData())
        X=xyz_arr[:,0]
        Y=xyz_arr[:,1]
        Z=xyz_arr[:,2]
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.scatter(X,Y,Z,c=d_radius,marker='o',cmap=plt.cm.jet)
        ax.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
          ax.plot([xb], [yb], [zb], 'w')        #ax=fig.add_subplot(223)
        #ax.plot(rad,kcden(csa))
        #ax=fig.add_subplot(224)
        #ax.scatter(v_radius,np.array(a_radius)/np.array(v_radius))
        ax.view_init(elev=40.,azim=40)
        fig.savefig(self.output_prefix+'_diaphragmPlot.png',dpi=180)
        #fig.show()

    
    def particle_radius_from_scale(self,sigma):
        return self.spacing*math.sqrt(2)*sigma
        
                                    

if __name__ == "__main__":
    desc = """Compute diaphram thickness phenotypes"""
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-i',help='VTK diaphragm particle filename', \
                      dest='dia_file',metavar='<string>',default=None)
    parser.add_option('-o',help='Output prefix name', \
                                    dest='output_prefix',metavar='<string>',default=None)
    parser.add_option('-s',help='Image spacing', \
                     dest='spacing',default=0.625)
    parser.add_option('-p', help='Enable plotting', dest='plot', \
                  action='store_true')

    (options,args) =  parser.parse_args()
    dt=DiaphragmThickness(options.dia_file,options.output_prefix,float(options.spacing),options.plot)
    dt.execute()
