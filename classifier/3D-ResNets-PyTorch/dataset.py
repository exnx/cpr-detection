from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5

# cpr project dataset
# from datasets.cpr_clip_dataset import VideoClipDataset

# for rate dataset
# from datasets.slow_clip_dataset import SlowClipDataset
# from datasets.slow_simple_dataset import SlowClipDataset  # deterministic, 0.5x or 1x
from datasets.slow_varied_dataset import SlowClipDataset  # deterministic, 0.4, 0.6, 0.8, 1.0 or 1.2x


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(
                    label_path,
                    video_id_path,
                    split_type,
                    frame_dir,
                    image_size=224,
                    window_size=48):

    training_data = SlowClipDataset(
                label_path,
                video_id_path,
                split_type,
                frame_dir,
                image_size,
                window_size)

    # training_data = VideoClipDataset(
    #             label_path,
    #             video_id_path,
    #             split_type,
    #             frame_dir,
    #             image_size)

    return training_data


def get_validation_data(
        label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size=224,
        window_size=48):

    validation_data = SlowClipDataset(
        label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size,
        window_size)

    # validation_data = VideoClipDataset(
    #     label_path,
    #     video_id_path,
    #     split_type,
    #     frame_dir,
    #     image_size)

    return validation_data, collate_fn

def get_inference_data(
        label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size=224,
        window_size=48,
        is_inference=True):

    inference_data = SlowClipDataset(
        label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size,
        window_size,
        is_inference=is_inference)  # get the video id in the label too

    # inference_data = VideoClipDataset(
    #     label_path,
    #     video_id_path,
    #     split_type,
    #     frame_dir,
    #     image_size)  # get the video id in the label too

    return inference_data, collate_fn
