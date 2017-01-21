#!/usr/bin/python

import matplotlib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

#Use Agg backend to allow non-interactive rendering 
#without relying on X-windows. Other options are: PDF, Cairo..
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VasculatureQualityControl():
  """Compute vasculature specific quality control images
    
    Paramters
    ---------
    
  """
  def __init__(self,vessel,output_prefix,plot=False):
    self._vessel=vessel
    self.output_prefix=output_prefix
  
  def execute(self):
    vessel=self._vessel
    array_v =dict();
            
    for ff in ["ChestRegion"]:
      tmp=vessel.GetPointData().GetArray(ff)
      if isinstance(tmp,vtk.vtkDataArray) == False:
        tmp=vessel.GetFieldData().GetArray(ff)
      array_v[ff]=vtk_to_numpy(tmp)

    xyz_arr=vtk_to_numpy(vessel.GetPoints().GetData())
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    X=xyz_arr[:,0]
    Y=xyz_arr[:,1]
    Z=xyz_arr[:,2]
    ax.scatter(X,Y,Z,s=1,c=array_v['ChestRegion'],marker='.',cmap=plt.cm.jet,linewidth=0)
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
      ax.plot([xb], [yb], [zb], 'w')
    ax.view_init(elev=20.,azim=80)
    fig.savefig(self.output_prefix+'_vasculatureQualityControl.png',dpi=180)
        

from optparse import OptionParser
if __name__ == "__main__":
    desc = """Compute vasculature phenotypes"""
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-v',help='VTK vessel particle filename',
                      dest='vessel_file',metavar='<string>',default=None)
    parser.add_option('-o',help='Output prefix name',
                                    dest='output_prefix',metavar='<string>',default=None)

    (options,args) =  parser.parse_args()
  
    readerV=vtk.vtkPolyDataReader()
    readerV.SetFileName(options.vessel_file)
    readerV.Update()
    vessel=readerV.GetOutput()

    vqc=VasculatureQualityControl(vessel,options.output_prefix)
    vqc.execute()
