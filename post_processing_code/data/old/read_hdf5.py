import json
import argparse
import os
import numpy as np
import h5py
import cv2
import sys
sys.path.insert(0,'..')  # for import from a directory 1 level up in fs
from utils.utils import read_json


def main(id_path, split, root):

    print('main called...')

    ext = '.hdf5'

    id_path_json = read_json(id_path)

    for id in id_path_json[split]:

        file_name = id + ext
        file_path = os.path.join(root, file_name)

        with h5py.File(file_path, 'r') as h5r:

            frames = h5r['images']

            import pdb; pdb.set_trace()

            # for cnt in range(10):
            #     print('get slice#', str(cnt))
            #     img_arr = h5r['Images'][cnt * 100:(cnt + 1) * 100]




if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id-path', help='path to video ids')
    parser.add_argument('-s', '--split', help='split type')
    parser.add_argument('-r', '--root', help='root to all hdf5 files')

    args = parser.parse_args()

    id_path = args.id_path
    split = args.split
    root = args.root

    main(id_path, split, root)

'''
python read_hdf5.py \
--id-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_cut_up.json \
--split train3 \
--root /vision2/u/enguyen/mini_cba/hdf5_fps10/train3

'''
