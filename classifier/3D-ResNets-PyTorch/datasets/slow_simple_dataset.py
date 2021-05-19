import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import argparse
import random
from PIL import Image
import time
import os


'''

- open clip metadata
- retrieve clip ids
    aggregate all into 1 list (instead of by video id)

in for loop
- get clip id
- get get total num frames from meta
- choose N start frames
    - need to sample somewhat evenly, and have at least 3T forward
- decide if slow or normal 50%, based on f = skip factor
- create frames list
    - have start and end (start + 3T frames)
    - skip a frame with prob 1 - 1/f
    - finalize frame list

'''


class SlowClipDataset(Dataset):
    def __init__(self, meta_path, video_id_path, split_type, frame_dir, image_size, window_size=48):

        self.set_seeds(0)  # set seeds

        self.EXT = '.jpeg'
        self.image_size = image_size
        self.frame_dir = frame_dir
        self.meta_data = self.read_json(meta_path)
        self.video_ids = self.read_json(video_id_path)[split_type]

        self.window_size = window_size  # number of frames to use
        self.space_between_starts = self.window_size * 2  # space between starts of compressions

        # returns a list of segments (start, end)
        # these are fixed
        self.segments = self.get_segments()

        # add transforms here
        self.video_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
            # transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])  # dan's
        ])

    def set_seeds(self, seed=0):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def read_json(self, path):
        with open(path) as f:
            data_dict = json.load(f)
        return data_dict

    def __len__(self):
        return len(self.segments)

    def get_segments(self):

        '''

        Given a video id, it calculates the candidate segments [start, end] to be used.



                                    space_between_starts
        begin of          <---------------------------------------->                                            end of video
        video
        ============================================================================================================
        =                 [start                 end]   ....       [start                 end]     ....            =
        =                 <----   window * 2   ---->                                                               =
        ============================================================================================================

        '''

        window_padded = self.window_size * 2  # we need room to speed (skip frames)

        start_delay = 0
        segments = []  # segments contain list of list [video_id, [start,end]]

        # loop thru video ids, then create segments and append to same list for all video ids
        for video_id in self.video_ids:

            num_frames = self.meta_data[video_id]['num_new_frames']

            # this ensures at least 1 segment included
            if num_frames < window_padded:
                segments.append([video_id, [0, num_frames]])

            for start in range(start_delay, num_frames, window_padded):

                # need to take make sure the segment is full
                if start + window_padded >= num_frames:
                    break

                end = start + window_padded
                segments.append([video_id, [start, end]])

        return segments

    def get_label(self, segment):

        '''

        segment: list (start, end) - used to see how many frames there are

        0 is slow down (0.5x speed)
        1 is normal version (range is 1x speed)

        Note: counter-intuitive:  videos are already slowed down by 0.5x.
        ie, so we need to speed up to make it normal

        '''

        start = segment[0]
        end = segment[1]
        num_frames = end - start

        # we need to ensure at least window_size num of frames are used
        if num_frames < self.window_size * 2:
            label = 0  # need to force it to be slow speed, so that it uses 24 frames
            skip_factor = 1
            return label, skip_factor
        else:
            # sample label
            label = random.choice([0, 1])

        # f_range are hyperparameters
        if label == 0:
            skip_factor = 1  # slow
        else:
            skip_factor = 2  # normal (fast)

        return label, skip_factor

    def extend_list(self, frame_list):

        # double the list by appending a reversed copy of the frame list
        reversed_list = frame_list[::-1]
        reversed_list.pop(0)  # need to remove very first before appending, since it's a duplicate of the last frame

        frame_list.extend(reversed_list)

        extended_list = frame_list[:self.window_size]

        return extended_list


    def get_frame_list(self, segment, skip_factor):

        '''

        This will retrieve the frame numbers between the segment (start, end) producing
        the desired label speed (slow or normal), by retrieving frames.  A frame
        will have probability p of dropping, which will produce the desired
        speed. p = 1 - 1/f, f = skip_factor

        '''

        start = segment[0]
        end = segment[1]

        frame_list = []

        for i in range(start, end, skip_factor):
            frame_list.append(i)

        final_list = frame_list[:self.window_size]

        if len(final_list) < self.window_size:

            print('!!!final_list is less than window!!!')
            print('start {}, end {}, f {}, len {}'.format(start, end, skip_factor, len(final_list)))
            print('frame list', frame_list)

            final_list = self.extend_list(final_list)

            print('Doubling the list! new list size:', len(final_list))

        return final_list


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

    def __getitem__(self, idx):

        video_id, segment = self.segments[idx]  # select clip

        # sample label and skip factor f
        # need to pass segment to make sure there's enough frames
        label, skip_factor = self.get_label(segment)
        frame_list = self.get_frame_list(segment, skip_factor)

        # retrieve frames
        images = self.get_images(video_id, frame_list)

        # # save images for debugging
        # save_frames(images, video_id, label)

        # import pdb; pdb.set_trace()

        # apply transforms
        img_tensors = [self.video_transform(img).unsqueeze(0) for img in images]

        # stack
        img_tensors = torch.cat(img_tensors, dim=0)
        label_tensor = torch.tensor(label)

        # change axis order, will now be C, T, W, H
        img_tensors = img_tensors.permute(1, 0, 2, 3)

        frame_list_np = np.asarray(frame_list)
        segment_np = np.asarray(segment)

        # just for debugging
        return [img_tensors, [label_tensor, video_id, segment_np, skip_factor, frame_list_np]]

        # return [img_tensors, [label_tensor, video_id, segment_np, skip_factor]]

def save_frames(frames, video_id, label):

    '''

    loop thru frames
        create path
        create dir
        save

    :param frames:
    :return:
    '''

    out_path = '/vision2/u/enguyen/results/slow_simple_frames_test_48/'

    dir = video_id + '_{}_'.format(label)
    dir_path = os.path.join(out_path, dir)
    os.makedirs(dir_path, exist_ok=True)

    for i in range(len(frames)):
        file_name = str(i).zfill(5) + '.jpeg'
        file_path = os.path.join(dir_path, file_name)
        img = frames[i]
        img.save(file_path)

    print('done saving frames:', video_id)

def main(args):
    meta_path = args.meta_path
    video_id_path = args.video_id
    split_type = args.split_type
    frame_dir = args.frame_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = SlowClipDataset(meta_path, video_id_path, split_type, frame_dir, image_size=224)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    count = 0

    data_time = AverageMeter()
    end_time = time.time()
    begin_time = time.time()

    for i, data in enumerate(train_dataloader, 0):
        count += 1
        frames, labels = data

        data_time.update(time.time() - end_time)

        label_tensor, video_id, segment, skip_factor, frame_list = labels
        end_time = time.time()

        # import pdb; pdb.set_trace()

        print('i: {} / {}, time: {data_time.val:.3f} ({data_time.avg:.3f})'.format(i, len(train_dataloader),
                                                                                   data_time=data_time))

    print('finished at: {:.3f}'.format(end_time - begin_time))


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


if __name__ == '__main__':
    # testing dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meta-path', help='path to meta data')
    parser.add_argument('-v', '--video-id', help='path to the json of all video ids by split')
    parser.add_argument('-s', '--split-type', help='train, val or test')
    parser.add_argument('-fd', '--frame-dir', help="dir to all the frames")
    parser.add_argument('-b', '--batch-size', type=int, help="batch size")
    parser.add_argument('-n', '--num-workers', type=int, help="num workers")

    args = parser.parse_args()

    main(args)

'''

# example usage

python slow_simple_dataset.py \
--meta-path /scr-ssd/enguyen/slowed_clips_0.5x/432_fps24/clip_metadata.json \
--video-id  /scr-ssd/enguyen/slowed_clips_0.5x/432_fps24/clip_ids_split_merged.json \
--split-type train \
--frame-dir /scr-ssd/enguyen/slowed_clips_0.5x/432_fps24/frames \
--batch-size 16 \
--num-workers 8

'''
