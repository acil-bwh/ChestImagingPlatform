import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class FilterParticlesBySize:
    def __init__(self):
        self._sigma0 = 1 / np.sqrt(2.) / 2.
        self._sigmap = 1 / np.sqrt(2.) / 2.

    def vessel_radius_from_sigma(self, scale, sp):
        spacing = np.prod(sp) ** (1 / 3.0)
        mask = scale < (2. / np.sqrt(2) * self._sigma0)
        rad = np.zeros(mask.shape)
        rad[mask] = np.sqrt(2.)*(np.sqrt((scale[mask]*spacing)**2.0 + (self._sigma0*spacing)**2.0) -0.5*self._sigma0*spacing)
        rad[~mask] = np.sqrt(2.)*(np.sqrt((scale[~mask]*spacing)**2.0 + (self._sigmap*spacing)**2.0) -0.5*self._sigmap*spacing)
        return rad

    def execute(self, input_pp, array_name, min_csa=None, max_csa=None):
        readerV = vtk.vtkPolyDataReader()
        readerV.SetFileName(input_pp)
        readerV.Update()
        vessel = readerV.GetOutput()

        tmp = vessel.GetPointData().GetArray(array_name)
        array_v = vtk_to_numpy(tmp)

        if array_name == 'scale':
            spacing = vtk_to_numpy(vessel.GetFieldData().GetArray("spacing"))
            vessel_radius = self.vessel_radius_from_sigma(array_v, spacing)
        else:
            vessel_radius = array_v

        p_csa = np.pi*vessel_radius**2

        if min_csa is None:
            min_csa = p_csa.min()
        if max_csa is None:
            max_csa = p_csa.max()

        out_pd = vtk.vtkPolyData()
        out_points = vtk.vtkPoints()
        pd_array_vec = list()
        for ii in range(vessel.GetPointData().GetNumberOfArrays()):
            array = vtk.vtkFloatArray()
            array.SetNumberOfComponents(vessel.GetPointData().GetArray(ii).GetNumberOfComponents())
            array.SetName(vessel.GetPointData().GetArray(ii).GetName())

            pd_array_vec.append(array)

        inc = 0
        for ii, nn in enumerate(p_csa):
            if min_csa <= nn < max_csa:
                out_points.InsertNextPoint(vessel.GetPoints().GetPoint(ii))

                for jj in range(vessel.GetPointData().GetNumberOfArrays()):
                    pd_array_vec[jj].InsertTuple(inc, vessel.GetPointData().GetArray(jj).GetTuple(ii))

                inc += 1

        out_pd.SetPoints(out_points)
        for jj in range(vessel.GetPointData().GetNumberOfArrays()):
            out_pd.GetPointData().AddArray(pd_array_vec[jj])

        for jj in range(vessel.GetFieldData().GetNumberOfArrays()):
            out_pd.GetFieldData().AddArray(vessel.GetFieldData().GetArray(jj))

        return out_pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Method to select a group of particles by cross-sectional area (CSA)')
    parser.add_argument('--in_pp', help='Input particle file (.vtk)', required=True, type=str, default=None)
    parser.add_argument('--array_name', help='Name of array to use for particle group selection. Options: scale | '
                                             'dnn_radius | dnn_vessel_lumen', required=True, type=str, default=None)
    parser.add_argument('--min_csa', help="Minimum cross-sectional area value. Particles outside [min_csa, max_csa) "
                                          "will be removed. If not specified min CSA will be computed from particles",
                        type=float, required=False, default=None)
    parser.add_argument('--max_csa', help="Maximum cross-sectional area value. Particles outside [min_csa, max_csa) "
                                          "will be removed. If not specified, max CSA will be computed from particles",
                        type=float, required=False, default=None)
    parser.add_argument('--o', help="Output particle file (.vtk)",
                        type=str, required=False, default=None)
    op = parser.parse_args()

    ff = FilterParticlesBySize()
    out_pp = ff.execute(op.in_pp, op.array_name, op.min_csa, op.max_csa)

    writer = vtk.vtkPolyDataWriter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(out_pp)
    else:
        writer.SetInputData(out_pp)
    writer.SetFileName(op.o)
    writer.SetFileTypeToBinary()
    writer.Update()
