#!/usr/bin/python

# import matplotlib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

from cip_python.common import ChestConventions


class VasculatureQualityControl():
  """Compute vasculature specific quality control images
    
    Paramters
    ---------
    
  """
  def __init__(self,vessel,output_prefix,plot=False):
    self._vessel=vessel
    self.output_prefix=output_prefix

  def get_artery_vein_indices(self, array_v):
      array_v = array_v.astype(np.uint16)
      type_values = ChestConventions.GetChestTypeFromValue(array_v)
      artery_type = ChestConventions.GetChestTypeValueFromName('Artery')
      vein_type = ChestConventions.GetChestTypeValueFromName('Vein')
      return np.argwhere(type_values == artery_type)[:, 0], np.argwhere(type_values == vein_type)[:, 0]

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
    vessel=self._vessel

    xyz_arr=vtk_to_numpy(vessel.GetPoints().GetData())
    
    array_v = dict()
    array_artery = dict()
    array_vein = dict()
    
    for ff in ["ChestRegionChestType","dnn_radius"]:
      tmp=vessel.GetPointData().GetArray(ff)
      if isinstance(tmp,vtk.vtkDataArray) == False:
        array_v[ff]=np.ones((xyz_arr.shape[0],))
      else:
        array_v[ff]=vtk_to_numpy(tmp)

    artery_idx, vein_idx = self.get_artery_vein_indices(array_v['ChestRegionChestType'])
    for ff in ["ChestRegionChestType", "dnn_radius"]:
        array_artery[ff] = array_v[ff][artery_idx]

    for ff in ["ChestRegionChestType", "dnn_radius"]:
        array_vein[ff] = array_v[ff][vein_idx]

    fig=plt.figure()

    # All
    all = fig.add_subplot(131, projection='3d', facecolor='white')
    X_artery=xyz_arr[artery_idx,0]
    Y_artery=xyz_arr[artery_idx,1]
    Z_artery=xyz_arr[artery_idx,2]
    X_vein=xyz_arr[vein_idx,0]
    Y_vein=xyz_arr[vein_idx,1]
    Z_vein=xyz_arr[vein_idx,2]
    all.scatter(X_artery, Y_artery, Z_artery, s=array_artery['dnn_radius'], c=array_artery['dnn_radius'], marker='.',
               cmap=plt.cm.Blues, linewidth=0, alpha=1)
    all.scatter(X_vein, Y_vein, Z_vein, s=array_vein['dnn_radius'], c=array_vein['dnn_radius'], marker='.',
               cmap=plt.cm.Reds, linewidth=0)

    self.plot_adjustments(all, xyz_arr, 'Artery-Vein')

    # Artery
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(X_artery,Y_artery,Z_artery,s=array_artery['dnn_radius'],c=array_artery['dnn_radius'],marker='.',cmap=plt.cm.Blues,linewidth=0)
    self.plot_adjustments(ax, xyz_arr[artery_idx], 'Artery')

    # Vein
    ay = fig.add_subplot(133, projection='3d')
    ay.scatter(X_vein,Y_vein,Z_vein,s=array_vein['dnn_radius'],c=array_vein['dnn_radius'],marker='.',cmap=plt.cm.Reds,linewidth=0)
    self.plot_adjustments(ay, xyz_arr[vein_idx], 'Vein')

    fig.savefig(self.output_prefix+'_vasculatureQualityControl.png',dpi=250, bbox_inches='tight')
        

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
