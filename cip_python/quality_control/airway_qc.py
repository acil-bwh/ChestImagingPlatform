#!/usr/bin/python

# import matplotlib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

from cip_python.common import ChestConventions


class AirwayQualityControl():
  """Compute airway specific quality control images
    
    Paramters
    ---------
    
  """
  def __init__(self,airway,output_prefix,plot=False):
    self._airway=airway
    self.output_prefix=output_prefix

  def get_airway_indices(self, array_a):
      array_a = array_a.astype(np.uint16)
      type_values = ChestConventions.GetChestTypeFromValue(array_a)
      airway_type = ChestConventions.GetChestTypeValueFromName('Airway')
      return np.argwhere(type_values == airway_type)[:, 0]

  def plot_adjustments(self, ax, xyz_arr, title, xlabel=None, ylabel=None, zlabel=None):
      ax.grid(False)

      ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
      ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
      ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

      ax.margins(0.0)
      if xlabel is not None:
          ax.set_xlabel(xlabel)
      if ylabel is not None:
          ax.set_ylabel(ylabel)
      if zlabel is not None:
          ax.set_zlabel(zlabel)

      plt.title(title)
      ax.view_init(elev=20., azim=80)

      ax.w_xaxis.set_ticks([])
      ax.w_yaxis.set_ticks([])
      ax.w_zaxis.set_ticks([])
      
      # Create cubic bounding box to simulate equal aspect ratio
      X = xyz_arr[:, 0]
      Y = xyz_arr[:, 1]
      Z = xyz_arr[:, 2]
      max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
      Xb_a = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
      Yb_a = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
      Zb_a = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

      # Comment or uncomment following both lines to test the fake bounding box:
      # for xb, yb, zb in zip(Xb_a, Yb_a, Zb_a):
      #     ax.plot([xb], [yb], [zb], 'w')

  def execute(self):
    airway=self._airway

    xyz_arr=vtk_to_numpy(airway.GetPoints().GetData())
    
    array_a = dict()
    array_airway = dict()
    
    for ff in ["ChestRegionChestType","dnn_lumen_radius"]:
      tmp=airway.GetPointData().GetArray(ff)
      if isinstance(tmp,vtk.vtkDataArray) == False:
        array_a[ff]=np.ones((xyz_arr.shape[0],))
      else:
        array_a[ff]=vtk_to_numpy(tmp)

    airway_idx = self.get_airway_indices(array_a['ChestRegionChestType'])
    for ff in ["ChestRegionChestType", "dnn_lumen_radius"]:
        array_airway[ff] = array_a[ff][airway_idx]

    fig=plt.figure()

    # All
    all = fig.add_subplot(131, projection='3d', facecolor='white')
    X_airway=xyz_arr[airway_idx,0]
    Y_airway=xyz_arr[airway_idx,1]
    Z_airway=xyz_arr[airway_idx,2]

    all.scatter(X_airway, Y_airway, Z_airway, s=array_airway['dnn_lumen_radius'], c=array_airway['dnn_lumen_radius'], marker='.',
               cmap=plt.cm.Blues, linewidth=0, alpha=1)


    self.plot_adjustments(all, xyz_arr, 'Airway')



    fig.savefig(self.output_prefix+'_airwayQualityControl.png',dpi=250, bbox_inches='tight')
        

from optparse import OptionParser
if __name__ == "__main__":
    desc = """Generate airway QC image """
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-v',help='VTK airway particle filename',
                      dest='airway_file',metavar='<string>',default=None)
    parser.add_option('-o',help='Output prefix name',
                                    dest='output_prefix',metavar='<string>',default=None)

    (options,args) =  parser.parse_args()
  
    readerV=vtk.vtkPolyDataReader()
    readerV.SetFileName(options.airway_file)
    readerV.Update()
    airway=readerV.GetOutput()

    aqc=AirwayQualityControl(airway,options.output_prefix)
    aqc.execute()
