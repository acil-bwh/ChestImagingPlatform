import argparse
import numpy as np
from cip_python.input_output import ImageReaderWriter

if __name__ == "__main__":
    descr='Method to generate weighted average image from dual energy CTs with linear blending'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--he', help="Input high energy CT file (.nrrd)", type=str, required=True)
    parser.add_argument('--le', help="Input low energy CT file (.nrrd)", type=str, required=True)
    parser.add_argument('--p', help="Percentage to use for low dose image. Default: 0.6", type=float, default=0.6,
                        required=True)
    parser.add_argument('--o', help="Output file (.nrrd)", type=str, required=True)

    op = parser.parse_args()

    io = ImageReaderWriter()
    he_ct, header = io.read_in_numpy(op.he)
    le_ct, _ = io.read_in_numpy(op.le)

    le_perc = float(op.p)
    he_perc = 1.0 - le_perc

    wa = (le_ct * le_perc) + (he_ct * he_perc)
    wa = np.round(wa).astype(np.int16)

    io.write_from_numpy(wa, metainfo=header, file_name=op.o)
