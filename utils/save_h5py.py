import os
import sys

import h5py
import mat73


def main():
    input_mat = os.environ.get('MAT73_INPUT_FILE')
    output_h5 = os.environ.get('H5_OUTPUT_FILE')
    dataset_key = os.environ.get('MAT73_DATASET_KEY', 'csm')

    if not input_mat or not output_h5:
        raise RuntimeError('Set MAT73_INPUT_FILE and H5_OUTPUT_FILE before running this utility.')

    data = mat73.loadmat(input_mat)[dataset_key]
    with h5py.File(output_h5, 'w') as h5_file:
        h5_file.create_dataset(dataset_key, data=data)


if __name__ == '__main__':
    sys.exit(main())
