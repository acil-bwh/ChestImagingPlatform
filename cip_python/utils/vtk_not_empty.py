import vtk
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to check if vtk file is empty')
    parser.add_argument('--in', dest='input_file', help="Input file", type=str, required=True)

    op = parser.parse_args()

    pp = vtk.vtkPolyDataReader()
    pp.SetFileName(op.input_file)
    pp.Update()

    if pp.GetOutput().GetNumberOfPoints() > 0:
        sys.exit(0)  # Returns True
    else:
        sys.exit(1)  # Returns False
