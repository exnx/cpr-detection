import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from PIL import Image
import numpy as numpy
import random
import json
import argparse
import math
import warnings
from torchvision import transforms



video_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])



class VideoClipDataset(Dataset):
    def __init__(self, annot_path, video_id_path, split_type, metadata_path, frame_dir, FPS=10, window_size=16):

        '''
        inputs:
            FPS, int: 
            annotation, str: 
            data_split, str: 
            window_size, int: 
            data_split_type, str:  train, val or test

        load annotation json

        load data split (train, val, or test)

        call create_window_labels()

        '''

        self.EXT = ".jpeg"
        self.FPS = FPS
        self.window_size = window_size
        self.frame_dir = frame_dir

        with open(annot_path) as f:
            self.annot_json = json.load(f)

        with open(video_id_path) as f:
            video_id = json.load(f)

        with open(metadata_path) as f:
            self.metadata_path = json.load(f)            

        # retrieve list of ids for the data split
        self.video_id_list = video_id[split_type]

        self.segments, self.labels = self._create_window_labels()

        return None

    def _get_start_end_idx(self, start_time, end_time):

        '''
        we need to make sure to choose the appropriate window given the start and end times.
        basically, times need to be in the majority of the window, otherwise we choose the 
        next window.  For start, that means +1 window.  For end, that means -1 window.

        '''

        # calc window idx the times correspond to
        start_idx = start_time / self.window_size
        
        # check start time
        if start_idx - math.floor(start_idx) > 0.9:
            start_idx = math.ceil(start_idx)
        else:
            start_idx = math.floor(start_idx)

        end_idx = end_time / self.window_size

        # check end time
        if end_idx - math.ceil(start_idx) < 0.1:
            end_idx = math.floor(end_idx)
        else:
            end_idx = math.ceil(end_idx)

        return start_idx, end_idx


    def _create_window_labels(self):

        '''

        lots of logic here.

        loop thru split, retrieve a video
            get length of a video
            calc number of frames total, round down nearest multiple of FPS
            create list of windows X, set all labels Y to 0 (no cpr)
            
            loop thru annotations
                get annot
                calc start/end frames, then corresponding idxs (for windows X)
                set window idxs to 1 (yes cpr)

        '''

        segments = []  # dict of lists, with each list containing frame num inside each window
        labels = []

        # loop thru each video in data split
        for video_id in self.video_id_list:
            # grab meta data for video
            metadata = self.metadata_path[video_id] # in secs
            duration = float(metadata['duration'])
            num_frames = int(duration * self.FPS)  # need to round down later... TODO

            # create window w/ 0 labels for each segment
            for i in range(0, num_frames, self.window_size):

                window = [i, i+self.window_size - 1]
                segments.append([video_id, window])
                labels.append(0)  # set false for cpr found
            
            # loop thru annotations, set label to 1 for windows w/ cpr
            for annot_id, annot in self.annot_json[video_id].items():

                if annot["valid"] == True:  # need to make sure it's valid first (this is how we track deleted)

                    start_time = annot['start_time']
                    end_time = annot['end_time']

                    start_frame = start_time * self.FPS
                    end_frame = end_time * self.FPS

                    start_idx, end_idx = self._get_start_end_idx(start_frame, end_frame)

                    for j in range(start_idx, end_idx):
                        labels[j] = 1  # set to true for cpr found

            # we remove the last window/label just to make sure the frames are all openable from disk
            segments.pop()
            labels.pop()

        return segments, labels

    def get_images(self, video_id, frame_list):

        images = []
        height = None
        width = None


        for i, frame_num in enumerate(frame_list):
            img = None

            frame_path = os.path.join(os.path.join(self.frame_dir, video_id), str(frame_num).zfill(5)) + self.EXT

            print('frame_path:', frame_path)

            if os.path.exists(frame_path):
                img = Image.open(frame_path)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                print('missing image:', frame_path)

            if img is not None and height is None:
                height = img.height
                width = img.width

        return images, height, width



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        video_id, segment = self.segments[idx]
        label = self.labels[idx]

        # create frame list
        frame_list = list(range(segment[0], segment[1]))

        # retrieve frames
        images, height, width = self.get_images(video_id, frame_list)

        # # transforms and augmentations
        # images = video_transform(images)

        img_tensors = [video_transform(img) for img in images]

        import pdb; pdb.set_trace()

        return img_tensors, height, width



def main(args):

    annot_path = args.annot
    video_id_path = args.video_id
    metadata_path = args.metadata
    split_type = args.split_type
    frame_dir = args.frame_dir
    FPS = 10
    window_size = 16

    dataset = VideoClipDataset(annot_path, video_id_path, split_type, metadata_path, frame_dir, FPS, window_size)
    images, height, width = next(iter(dataset))

    print('len images', len(images))


if __name__ == '__main__':
    # testing dataset


    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annot', help='path to the annot file')
    parser.add_argument('-v', '--video-id', help='path to the json of all video ids by split')
    parser.add_argument('-s', '--split-type', help='train, val or test')
    parser.add_argument('-m', '--metadata', help='path to the video metadata')
    parser.add_argument('-fd', '--frame-dir', help="dir to all the frames")

    args = parser.parse_args()

    main(args)



'''

python video_dataset.py \
--annot ~/Desktop/webtool_data_apr8/annotation_info.json \
--video-id ~/Desktop/cpr-detection/train_data.json \
--split-type train \
--metadata ~/Desktop/cpr-detection/video_metadata.json \
--frame-dir ~/Desktop/frames/








'''
