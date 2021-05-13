import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from PIL import Image
import random
import json
import argparse
import math
import warnings
from torchvision import transforms

import sys
sys.path.insert(0,'..')  # for import from a directory 1 level up in fs
from utils.utils import write_json, read_json


class VideoClipDataset(Dataset):
    def __init__(self, segment_label_path, video_id_path, split_type, frame_dir):

        '''

        Uses a segment label json to retrieve images in a window, and its label

        segment = [video_id, [start,end]], where start / end are the interval of frames

        :param segment_label_path: str, path to json containing segments and labels
        :param video_id_path: str, path to json with video id splits
        :param split_type: str, train val or test
        :param frame_dir: str, path to the frames root
        '''

        self.EXT = ".jpeg"
        self.frame_dir = frame_dir
        self.image_height = 224
        self.image_width = 224

        # retrieve list of ids for the data split
        video_id_by_split = read_json(video_id_path)
        self.video_id_list = video_id_by_split[split_type]

        # retrieve segment and labels
        segments_and_labels = read_json(segment_label_path)[split_type]
        self.segments = segments_and_labels['segments']
        self.labels = segments_and_labels['labels']

        # add transforms here
        self.video_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
            # transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])  # dan's
        ])


    def get_images(self, video_id, frame_list):

        '''

        :param video_id: str
        :param frame_list: list, of strings that contain the frame numbers for a segment
        :return:
            images: list of PIL images
            height: int
            width: int
        '''

        images = []
        height = None
        width = None

        for i, frame_num in enumerate(frame_list):
            img = None

            frame_path = os.path.join(os.path.join(self.frame_dir, video_id), str(frame_num).zfill(5)) + self.EXT
            # print('frame_path:', frame_path)

            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                images.append(img)
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

        '''

        Given a segment idx, retrieve the segment frames nums and video id,
        then get the corresponding frames. (as well as label)
        Apply transforms, convert to tensors.

        :param idx: random video idx
        :return:
            image_tensors: tensor, shape B, T, C, H, W, where T = window size
            labels: tensor
        '''

        video_id, segment = self.segments[idx]
        label = self.labels[idx]

        # create frame list
        frame_list = list(range(segment[0], segment[1] + 1))  # need to add 1 to make inclusive

        # retrieve frames
        images, height, width = self.get_images(video_id, frame_list)

        # apply transforms
        img_tensors = [self.video_transform(img).unsqueeze(0) for img in images]

        # stack
        img_tensors = torch.cat(img_tensors, dim=0)
        label_tensor = torch.tensor(label)

        # change axis order
        img_tensors = img_tensors.permute(1, 0, 2, 3)
        
        return img_tensors, label_tensor



def main(args):

    segment_label_path = args.segment_label
    video_id_path = args.video_id
    split_type = args.split_type
    frame_dir = args.frame_dir
    batch_size = args.batch_size

    dataset = VideoClipDataset(segment_label_path, video_id_path, split_type, frame_dir)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, data in enumerate(train_dataloader, 0):
        frames, label = data
        import pdb;
        pdb.set_trace()


if __name__ == '__main__':
    # testing dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-sg', '--segment-label', help='segment and label path')
    parser.add_argument('-v', '--video-id', help='path to the json of all video ids by split')
    parser.add_argument('-s', '--split-type', help='train, val or test')
    parser.add_argument('-fd', '--frame-dir', help="dir to all the frames")
    parser.add_argument('-b', '--batch-size', type=int, help="batch size")

    args = parser.parse_args()

    main(args)



'''

# example usage

python video_dataset.py \
--segment-label /vision2/u/enguyen/cpr-detection/post_processing_code/data/segments_and_labels.json \
--video-id /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_split.json \
--split-type train \
--frame-dir /vision2/u/enguyen/mini_cba/new_fps10 \
--batch-size 8

'''
