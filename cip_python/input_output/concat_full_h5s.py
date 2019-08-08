import argparse
import h5py

from cip_python.input_output import H5DatasetStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to concat a series of h5 files fully filled (same schema) '
                                                 'in a single h5. Key names and keys description will be copied from'
                                                 'the first h5 in the specified paths list.')
    parser.add_argument('--descr', dest='h5_file_description', help="Description of the final h5", type=str,
                        required=True)
    parser.add_argument('--paths_list', dest='h5_paths_list', help=".txt file containing the full paths to the single "
                                                                   "h5 files to merge",
                        type=str,
                        required=True)
    parser.add_argument('--override', dest='override_if_existing',
                        help="If specified, the existence of a previous h5 file will be ignored and it will be "
                             "overwritten", action='store_true')
    parser.add_argument('--o', dest='out_h5', help="Output h5 file", required=True, type=str)

    op = parser.parse_args()
    paths_list = open(op.h5_paths_list, "r").read().split("\n")[:-1]
    with h5py.File(paths_list[0], 'r') as smh5f:
        key_names = tuple(smh5f.attrs['key_names'].split(';'))
        keys_description = smh5f.attrs['keys_description']

    H5DatasetStore.concat_full_h5s_class(paths_list, str(op.h5_file_description), op.out_h5,
                                         key_names=key_names,
                                         keys_description=keys_description,
                                         override_if_existing=op.override_if_existing)
