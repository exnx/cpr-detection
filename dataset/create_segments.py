import os
import numpy as numpy
import random
import json
import argparse
import math

import sys
sys.path.insert(0,'..')  # for import from a directory 1 level up in fs
from utils.utils import write_json, read_json


class Segments:
    def __init__(self, annot_path, video_id_path, metadata_path, fps, window_size, out_path):
        '''

        Given videos, we create segments (of frames) and their corresponding labels.
        A segment is a start/end frame numbers (for a video) and label is whether compression
        occurs or not in that segment.  We use the annotations in secs of the videos to calc
        the label.

        :param annot_path: str, path to the cpr annotations
        :param video_id_path: str, path for the video ids by train/val/test splits
        :param metadata_path: str, path to the metadata of the videos
        :param fps: int, fps of frames videos were converted to
        :param window_size: int, num of frames in a sliding window
        :param out_path: str, path to output the segment and labels json

        '''

        self.fps = fps
        self.window_size = window_size

        self.annot_json = read_json(annot_path)
        video_id_by_split = read_json(video_id_path)
        self.metadata_path = read_json(metadata_path)

        # store each split here
        all_data = {}

        # loop thru each data split
        for split_type in video_id_by_split.keys():
            video_id_list = video_id_by_split[split_type] # retrieve a video id
            segments, labels = self._create_segments_labels(video_id_list)  # create the segments/labels
            data = {'segments': segments, 'labels': labels}  # store both in a dict
            all_data[split_type] = data  # store for entire video

        # write all to disk
        out_path = os.path.join(out_path, 'segments_and_labels.json')
        write_json(all_data, out_path, indent=None)

    def _create_segments_labels(self, video_id_list):

        '''

        Assuming we have N frames for a video, we create sliding window of segments and their
        corresponding cpr label.

        :param video_id_list: list, of video ids for a given split
        :return:
            segment: list - [video_id, [start, end], ... ]
            labels: list -  [0, 0, 1, 0 ... 0, 1]

        # rough logic
        loop thru each video

            get length of a video, num_new_frames

            loop thru and create list of windows X, set all labels Y to 0 (no cpr)

            loop thru annotations (2nd for loop)
                get annot
                calc start/end frames, then corresponding idxs (for windows X)
                set window idxs to 1 (yes cpr)

        '''

        all_segments = []  # dict of lists, with each list containing frame num inside each window
        all_labels = []

        # loop thru each video in data split
        for video_id in video_id_list:

            # create segments / labels for this single video id
            single_segments = []
            single_labels = []

            # grab meta data for video
            metadata = self.metadata_path[video_id] # in secs
            duration = float(metadata['duration'])
            # num_frames = int(duration * self.fps)  # need to round down later... TODO
            num_frames = int(metadata['num_new_frames'])  # we now have actual num of frames after fps sampling

            # create window w/ 0 labels for each segment
            for i in range(0, num_frames, self.window_size):

                window = [i, i+self.window_size - 1]
                single_segments.append([video_id, window])
                single_labels.append(0)  # set false for cpr found

            # loop thru annotations, set label to 1 for windows w/ cpr
            for annot_id, annot in self.annot_json[video_id].items():

                if annot["valid"] == True:  # need to make sure it's valid first (this is how we track deleted)

                    start_time = annot['start_time']
                    end_time = annot['end_time']

                    start_frame = start_time * self.fps
                    end_frame = end_time * self.fps

                    start_idx, end_idx = self._get_start_end_idx(start_frame, end_frame)

                    end_idx = min(end_idx, len(single_segments)-1)

                    if end_idx < start_idx:
                        end_idx = start_idx

                    try:
                        for j in range(start_idx, end_idx+1):  # need to make inclusive
                            single_labels[j] = 1  # set to true for cpr found

                    except:
                        import pdb;
                        pdb.set_trace()

            # we remove the last window/label just to make sure the frames are all openable from disk
            single_segments.pop()
            single_labels.pop()

            # add all to the ongoing list of segments and labels
            all_segments.extend(single_segments)
            all_labels.extend(single_labels)
            # print('video_id {}, label sum: {}'.format(video_id, sum(single_labels)))

        return all_segments, all_labels


    def _get_start_end_idx(self, start_time, end_time):

        '''

        Given times, we need to convert to start-end time intervals to
        their corresponding index / segment number (for where compression occurs).

        Also, times need to be in the majority (or a threshold) of the window, otherwise we choose the
        next window.  For start, that means +1 window.  For end, that means -1 window.

        :param start_time: float, time compression starts in a video
        :param end_time: float, time compression end in a video
        :return:
        '''

        start_thresh = 0.9
        end_thresh = 0.1

        # calc window idx the times correspond to
        start_idx = start_time / self.window_size

        # check start time
        if start_idx - math.floor(start_idx) > start_thresh:
            start_idx = math.ceil(start_idx)
        else:
            start_idx = math.floor(start_idx)

        end_idx = end_time / self.window_size

        # check end time
        if end_idx - math.ceil(start_idx) < end_thresh:
            end_idx = math.floor(end_idx)
        else:
            end_idx = math.ceil(end_idx)

        return start_idx, end_idx


def main(args):

    annot_path = args.annot
    video_id_path = args.video_id
    metadata_path = args.metadata
    fps = args.fps
    window_size = args.window_size
    out_path = args.out_path

    segments = Segments(annot_path, video_id_path, metadata_path, fps, window_size, out_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annot', help='path to the annot file')
    parser.add_argument('-v', '--video-id', help='path to the json of all video ids by split')
    parser.add_argument('-m', '--metadata', help='path to the video metadata')
    parser.add_argument('-f', '--fps', default=10, type=int, help='FPS of videos to frames created')
    parser.add_argument('-w', '--window-size', default=16, type=int, help='sliding window size of frames')
    parser.add_argument('-o', '--out-path', help='output path')

    args = parser.parse_args()

    main(args)



'''

python create_segments.py \
--annot /vision2/u/enguyen/cpr-detection/post_processing_code/data/annotation_info.json \
--video-id /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_split.json \
--metadata /vision2/u/enguyen/cpr-detection/post_processing_code/data/video_metadata.json \
--out-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/

'''
