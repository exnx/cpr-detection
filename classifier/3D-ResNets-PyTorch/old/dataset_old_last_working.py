from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5

# cpr project dataset
from datasets.cpr_clip_dataset import VideoClipDataset


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(
                    segment_label_path,
                    video_id_path,
                    split_type,
                    frame_dir,
                    image_size=224):

    training_data = VideoClipDataset(
                segment_label_path,
                video_id_path,
                split_type,
                frame_dir,
                image_size)

    return training_data


def get_validation_data(
        segment_label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size=224):

    validation_data = VideoClipDataset(
        segment_label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size)

    return validation_data, collate_fn


def get_inference_data(
        segment_label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size=224):

    inference_data = VideoClipDataset(
        segment_label_path,
        video_id_path,
        split_type,
        frame_dir,
        image_size)  # get the video id in the label too

    return inference_data, collate_fn
