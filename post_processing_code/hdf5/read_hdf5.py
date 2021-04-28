import json
import argparse
import os
import numpy as np
import h5py
import cv2

import sys
sys.path.insert(0,'..')  # for import from a directory 1 level up in fs
from utils.utils import read_json




def find_means(video_means):
    import pdb;
    pdb.set_trace()

    vals = np.array([np.array(xi) for xi in video_means])
    # vals = mean0, mean1, mean2, num_frames

    # sum_frames = np.sum(vals, axis=0)[3]
    sum_frames = np.sum(vals[:, 3])
    weights = val[:, 3] / sum_frames

    avg = np.average(vals, axis=0, weights=weights)
    return avg

def main(id_path, split, meta_path, root, chunk_size):
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

    ext = '.hdf5'
    id_json = read_json(id_path)
    meta_json = read_json(meta_path)

    video_means = []

    # for split in id_json.keys():
    for i, f_id in enumerate(id_json[split]):

        file_name = f_id + ext
        file_path = os.path.join(root, file_name)
        num_frames = meta_json[f_id]['num_new_frames']

        mean0 = 0
        mean1 = 0
        mean2 = 0

        with h5py.File(file_path, 'r') as h5r:

            import pdb;
            pdb.set_trace()

            # slicing against the grain
            # mean0 = np.mean(h5r.get('images')[:, :, :, 0])
            # mean1 = np.mean(h5r.get('images')[:, :, :, 1])
            # mean2 = np.mean(h5r.get('images')[:, :, :, 2])

            count = 0

            # loop thru, 1 chunk at a time
            for f_num in range(0, num_frames, chunk_size):
                chunk_frames = h5r.get('images')[f_num:f_num+chunk_size]
                mean0 += np.mean(chunk_frames[:, :, :, 0])
                mean1 += np.mean(chunk_frames[:, :, :, 1])
                mean2 += np.mean(chunk_frames[:, :, :, 2])
                count += 1

        # import pdb; pdb.set_trace()

        # need to divide by the number of chunks
        mean0 = mean0 / count / 255
        mean1 = mean1 / count / 255
        mean2 = mean2 / count / 255


        print('video id:', f_id)
        video_means.append([mean0, mean1, mean2, num_frames])
        print(video_means)

        if i >= 3:
            break
            # import pdb; pdb.set_trace()

    find_means(video_means)

            # print('frame shame {}'.format(f_id))
            # print(frames.shape)





if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id-path', help='path to video ids')
    parser.add_argument('-s', '--split', help='split type')
    parser.add_argument('-m', '--meta-path', help='path to meta data output')
    parser.add_argument('-r', '--root', help='path to frames dir')
    parser.add_argument('-c', '--chunk-size', type=int, help='size of chunks (num of frames)')

    args = parser.parse_args()

    id_path = args.id_path
    split = args.split
    meta_path = args.meta_path
    root = args.root
    chunk_size = args.chunk_size

    main(id_path, split, meta_path, root, chunk_size)


'''
python read_hdf5.py \
--id-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_cut_up.json \
--split train3 \
--meta-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/video_metadata.json \
--root /vision2/u/enguyen/mini_cba/hdf5_fps10/train3 \
--chunk-size 16

'''