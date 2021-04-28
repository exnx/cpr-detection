import json
import argparse
import os
import numpy as np
import h5py
import cv2

import sys
sys.path.insert(0,'..')  # for import from a directory 1 level up in fs
from utils.utils import read_json


def get_shape(height, width, max_height=360):

    '''

    Retrieve the downsized height and width.  This is same logic
    used to downsize beforehand.

    :param height:
    :param width:
    :return:
        new_height, int
        new_width, int
    '''

    aspect = width / height

    if height > max_height:
        new_height = 360
    else:
        return height, width

    new_width = int(new_height * aspect)

    return new_height, new_width


def main(id_path, split, frame_dir, meta_path, out_path, chunk_size=16):
    '''

    :param id_path:
    :param frame_path:
    :param meta_path:
    :param out_path:
    :param chunk_size:
    :return:

    # loop thru ids
        # retrieve num frames from metadata
        #  get shape
        # stick in h5 code

    '''

    out_path = os.path.join(out_path, split)
    os.makedirs(out_path, exist_ok=True)

    C = 3
    T = chunk_size
    ext = '.jpeg'

    meta_json = read_json(meta_path)
    id_json = read_json(id_path)

    # for split in id_json.keys():
    for id in id_json[split]:

        frame_path = os.path.join(frame_dir, id)

        # retrieve metadata
        num_frames = meta_json[id]['num_new_frames']
        old_height = meta_json[id]['height']
        old_width = meta_json[id]['width']
        height, width = get_shape(old_height, old_width)

        # create a new h5 file
        h5_out_path = os.path.join(out_path, '{}.hdf5'.format(id))
        with h5py.File(h5_out_path, 'w') as h5w:

            img_ds = h5w.create_dataset('images', shape=(num_frames+T, width, height, C), dtype=np.uint8,
                                        chunks=(T, width, height, C), compression="gzip", compression_opts=9, scaleoffset=1)
            # loop thru all images
            for i in range(0, num_frames, T):
                t_images = np.zeros((T, width, height, C))

                print('i:', i)

                # open T at a time
                for j in range(T):
                    num = i + j
                    img_name = str(num).zfill(5) + ext
                    img_path = os.path.join(frame_path, img_name)

                    # # try read as binary
                    # with open(img_path, mode='rb') as file:  # b is important -> binary
                    #     fileContent = file.read()
                    #
                    # import pdb; pdb.set_trace()

                    # img_np = np.fromfile(fileContent, dtype=uint8)

                    img_np = cv2.imread(img_path)  # cv2 method

                    t_images[j, :, :, :] = img_np

                # write chunk
                top = i + T
                h5w['images'][i:top] = t_images

                if i > 200:
                    break

                # import pdb; pdb.set_trace()


if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id-path', help='path to video ids')
    parser.add_argument('-s', '--split', help='split type')
    parser.add_argument('-f', '--frame-path', help='path to frames dir')
    parser.add_argument('-m', '--meta-path', help='path to meta data output')
    parser.add_argument('-o', '--out-path', help='path to hdf5 output')
    parser.add_argument('-c', '--chunk-size', type=int, help='size of chunks (num of frames)')

    args = parser.parse_args()

    id_path = args.id_path
    split = args.split
    frame_path = args.frame_path
    meta_path = args.meta_path
    out_path = args.out_path
    chunk_size = args.chunk_size

    main(id_path, split, frame_path, meta_path, out_path, chunk_size)

'''
python frames_to_hdf5.py \
--id-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_cut_up.json \
--split val \
--frame-path /vision2/u/enguyen/mini_cba/new_fps10 \
--meta-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/video_metadata.json \
--out-path /vision2/u/enguyen/mini_cba/hdf5_trial3/ \
--chunk-size 16

'''