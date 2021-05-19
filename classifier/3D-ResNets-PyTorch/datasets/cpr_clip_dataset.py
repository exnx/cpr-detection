import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from PIL import Image
# import numpy as np
import random
import json
import argparse
import math
import warnings
from torchvision import transforms
import time


class VideoClipDataset(Dataset):
    def __init__(self, segment_label_path, video_id_path, split_type, frame_dir, size=224):

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
        self.image_height = size
        self.image_width = size

        # retrieve list of ids for the data split
        video_id_by_split = self.read_json(video_id_path)
        self.video_id_list = video_id_by_split[split_type]

        # retrieve segment and labels
        segments_and_labels = self.read_json(segment_label_path)[split_type]
        self.segments = segments_and_labels['segments']
        self.labels = segments_and_labels['labels']

        # add transforms here
        self.video_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
            # transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])  # dan's
        ])

        # self.class_names = {'0': "None", '1': 'compression'}
        self.class_names = ["None", "compression"]

    def read_json(self, path):

        try:
            with open(path) as f:
                data_dict = json.load(f)

        except Exception as e:
            print(e)
            raise Exception

        return data_dict


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

        for i, frame_num in enumerate(frame_list):

            frame_path = os.path.join(os.path.join(self.frame_dir, video_id), str(frame_num).zfill(5)) + self.EXT

            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                images.append(img)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                print('missing image:', frame_path)

        return images

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
        images = self.get_images(video_id, frame_list)

        # apply transforms
        img_tensors = [self.video_transform(img).unsqueeze(0) for img in images]

        # stack
        img_tensors = torch.cat(img_tensors, dim=0)
        label_tensor = torch.tensor(label)

        # change axis order
        img_tensors = img_tensors.permute(1, 0, 2, 3)
        
        return [img_tensors, [label_tensor, video_id, segment, None]]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):

    segment_label_path = args.segment_label
    video_id_path = args.video_id
    split_type = args.split_type
    frame_dir = args.frame_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = VideoClipDataset(segment_label_path, video_id_path, split_type, frame_dir)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    # batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()

    begin_time = time.time()

    for i, data in enumerate(train_dataloader):

        if i == 1000:
            break

        data_time.update(time.time() - end_time)

        frames, labels = data
        video_ids, targets = labels
        end_time = time.time()

        print('i: {} / {}, time: {data_time.val:.3f} ({data_time.avg:.3f})'.format(i, len(train_dataloader), data_time=data_time))

        # import pdb;
        # pdb.set_trace()

    print('finished at: {:.3f}'.format(end_time - begin_time))


if __name__ == '__main__':
    # testing dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-sg', '--segment-label', help='segment and label path')
    parser.add_argument('-v', '--video-id', help='path to the json of all video ids by split')
    parser.add_argument('-s', '--split-type', help='train, val or test')
    parser.add_argument('-fd', '--frame-dir', help="dir to all the frames")
    parser.add_argument('-b', '--batch-size', type=int, help="batch size")
    parser.add_argument('-n', '--num-workers', type=int, help="num workers")

    args = parser.parse_args()

    main(args)



'''

# example usage

python cpr_clip_dataset.py \
--segment-label /vision2/u/enguyen/cpr-detection/post_processing_code/data/segments_and_labels.json \
--video-id /vision2/u/enguyen/cpr-detection/post_processing_code/data/data_split.json \
--split-type train \
--frame-dir /vision2/u/enguyen/mini_cba/new_fps10 \
--batch-size 32 \
--num-workers 28

'''
