import argparse
import numpy as np
from skimage.morphology import skeletonize

from cip_python.input_output.image_reader_writer import ImageReaderWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to reduce label map to 1 pixel wide representations')
    parser.add_argument('--i_lm', help="Path to labelmap (.nrrd)", type=str, required=True)
    parser.add_argument('--o', help="Path to save skeletonized labelmap (.nrrd)", type=str, required=True)
    args = parser.parse_args()

    io = ImageReaderWriter()

    in_lm, metainfo = io.read_in_numpy(args.i_lm)
    in_lm[in_lm > 1] = 1
    out_lm = skeletonize(in_lm)

    io.write_from_numpy(out_lm, metainfo, args.o)





