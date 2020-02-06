#!/usr/bin/python

import matplotlib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BronchialLumenQualityControl():
    """Compute bronchial tree specific quality control images

      Paramters
      ---------

    """

    def __init__(self, airway, output_prefix):
        self._airway = airway
        self.output_prefix = output_prefix

    def execute(self):
        airway = self._airway

        xyz_arr = vtk_to_numpy(airway.GetPoints().GetData())

        array_a = dict();

        for ff in ["ChestRegionChestType", "dnn_lumen_radius"]:
            tmp = airway.GetPointData().GetArray(ff)
            if isinstance(tmp, vtk.vtkDataArray) == False:
                array_a[ff] = np.ones((xyz_arr.shape[0],))
            else:
                array_a[ff] = vtk_to_numpy(tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        X = xyz_arr[:, 0]
        Y = xyz_arr[:, 1]
        Z = xyz_arr[:, 2]
        ax.scatter(X, Y, Z, s=array_a['dnn_lumen_radius'], c=array_a['dnn_lumen_radius'], marker='.', cmap=plt.cm.jet,
                   linewidth=0)
        ax.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        ax.view_init(elev=20., azim=80)
        fig.savefig(self.output_prefix + '_bronchialTreeQualityControl.png', dpi=180)


from optparse import OptionParser

if __name__ == "__main__":
    desc = """Compute airway tree QC image"""

    parser = OptionParser(description=desc)
    parser.add_option('-a', help='VTK airway particle filename',
                      dest='airway_file', metavar='<string>', default=None)
    parser.add_option('-o', help='Output prefix name',
                      dest='output_prefix', metavar='<string>', default=None)

    (options, args) = parser.parse_args()

    readerA = vtk.vtkPolyDataReader()
    readerA.SetFileName(options.airway_file)
    readerA.Update()
    airway = readerA.GetOutput()

    aqc = BronchialLumenQualityControl(airway, options.output_prefix)
    aqc.execute()
