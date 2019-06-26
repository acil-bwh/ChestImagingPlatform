import vtk


def merge_particle_files(input_particles):
    """
    Given a list of particle files, it merges all of them together. If a file does not contain vertices, they are added.
    :param input_particles: list of polydatas to merge
    :return: a vtk file containing all the merged polydata
    """
    aa = vtk.vtkAppendPolyData()

    for pp in input_particles:
        rr = vtk.vtkPolyDataReader()
        rr.SetFileName(pp)
        rr.Update()

        pd = rr.GetOutput()

        if pd.GetNumberOfCells() == 0:
            cc = vtk.vtkCellArray()
            for pid in range(pd.GetNumberOfPoints()):
                v = vtk.vtkVertex()
                v.GetPointIds().SetId(0, pid)
                cc.InsertNextCell(v)
            pd.SetVerts(cc)

        aa.AddInputData(pd)

    aa.Update()
    return aa.GetOutput()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VTK method to merge particles polydata')
    parser.add_argument('--i', nargs='*', dest='particles', help='Input particle files to merge. Multiple values '
                                                                 'allowed.')
    parser.add_argument('--o', dest='out_file', help='Output file name.')

    op = parser.parse_args()

    merged_file = merge_particle_files(op.particles)

    ww = vtk.vtkPolyDataWriter()
    ww.SetInputData(merged_file)
    ww.SetFileName(op.out_file)
    ww.Update()
