import json
import argparse
import os
import ffmpeg
import copy

'''

this will loop thru data splits and all video ids, get the meta data, convert to frames,
count the number of frames made. Will log any errors (video ids) at any stage.

'''


def count_num_frames(input_dir):

    '''
    given dir, count the number of files inside

    :param input_dir:
    :return:
    '''

    for root, subdirectories, files in os.walk(input_dir):
        num_files = len(files)
        # print('{}: num files {}'.format(root, num_files))

    return num_files


def main(video_ids_path, path_to_raw_videos, meta_path, frame_path, start_idx, end_idx, new_fps):

    '''

    Convert video to frames, and gather the meta data.  We also need to track
    anywhere in the process it fails, such as missing videos, can't open videos,
    can't convert videos with ffmpeg, or if the incorrect number of frames were
    created.  We log the video ids for any that fail at any of these steps.


    :param datasplit_path: dict, train, val, test splits, each with a list of ids
    :param path_to_raw_videos: str, location of videos
    :param meta_path: str, path to the meta data of the videos
    :param frame_path: str, location of frames to output
    :return: None
    '''

    # hard code this
    new_height = 224
    new_width = 224

    print('Processing start: {} to end: {}'.format(start_idx, end_idx))

    os.makedirs(meta_path, exist_ok=True)

    metadata = {}
    missing_list = []
    conversion_fail = []
    count_off = []

    with open(video_ids_path) as f:
        video_ids_list = json.load(f)['clip_filenames']

    video_metadata = {}

    # loop thru ids
    for i, video_id in enumerate(video_ids_list):

        # make sure we only process idxs from [start:end]
        if start_idx is not None:
            if i < start_idx or i >= end_idx:
                continue

        print('video id', video_id)
        # create video path
        video_path = os.path.join(path_to_raw_videos, "{}.mp4".format(video_id))

        # try opening and getting metadata
        try:
            # retrieve meta data
            meta_data = ffmpeg.probe(video_path)['streams'][0]
            video_metadata['duration'] = meta_data['duration']
            video_metadata['height'] = meta_data['height']
            video_metadata['width'] = meta_data['width']
            video_metadata['num_orig_frames'] = meta_data['nb_frames']
            video_metadata['avg_frame_rate'] = meta_data['avg_frame_rate']

        except ffmpeg.Error as e:
            print('video id not found:', video_id)
            print(e.stderr)
            missing_list.append(video_id)

            print('current missing')
            for i in missing_list:
                print(i)

        # try converting videos to frames
        try:
            # need to convert to frames
            frame_dir = os.path.join(frame_path, video_id)
            os.makedirs(frame_dir, exist_ok=True)

            # use python wrapper for ffmpeg (more stable than ffmpeg in terminal)
            (ffmpeg.input(video_path)
             .filter('fps', fps=new_fps)  # use all frames this time
             .output(os.path.join(frame_dir, "%5d.jpeg"),
                     video_bitrate='5000k',
                     s='{}x{}'.format(str(new_width), str(new_height)),
                     sws_flags='bilinear',
                     # **{'qscale:v': 3},  # not sure how to use this quality arg
                     start_number=0)
             .run(capture_stdout=True, capture_stderr=True))

            # update new meta_data
            video_metadata['new_height'] = new_height
            video_metadata['new_width'] = new_width
            video_metadata['new_fps'] = new_fps

        except ffmpeg.Error as e:
            print('video conversion failed to start:', video_id)
            print(e.stderr)
            conversion_fail.append(video_id)

            print('current conv. failed')
            for i in conversion_fail:
                print(i)

        # save metadata
        if video_metadata != {}:
            # compare orig to new number of frames, make note if different
            new_frame_count = count_num_frames(frame_dir)

            diff = abs(new_frame_count - float(meta_data['duration']) * new_fps)
            print('count {}, duration {}, diff {:.2f}, old fps {}'.format(new_frame_count, float(meta_data['duration']), diff, video_metadata['avg_frame_rate']))
            if diff > new_fps:
                count_off.append(video_id)

                print('current off count')
                for i in count_off:
                    print(i)

            video_metadata['num_new_frames'] = new_frame_count
            video_metadata['new_duration'] = new_frame_count / new_fps
            metadata[video_id] = copy.deepcopy(video_metadata)

    meta_out = os.path.join(meta_path, "video_metadata.json")
    missing_path = os.path.join(meta_path, 'missing_list.txt')
    conversion_fail_path = os.path.join(meta_path, 'conversion_fail.txt')
    count_off_path = os.path.join(meta_path, 'count_off.txt')

    # write metadata to disk
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # write missing files to disk
    with open(missing_path, "w") as f:
        for item in missing_list:
            f.write("%s\n" % item)

    # conversion fail tracking
    with open(conversion_fail_path, "w") as f:
        for item in conversion_fail:
            f.write("%s\n" % item)

    # write video ids that have different number of frames than duration
    with open(count_off_path, "w") as f:
        for item in count_off:
            f.write("%s\n" % item)


if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--video-ids', help='path to video ids')
    parser.add_argument('-rv', '--raw', help='path to raw videos')
    parser.add_argument('-m', '--meta-path', default=None, help='path to meta data output')
    parser.add_argument('-f', '--frame-path', default=None, help='path to frames output')
    parser.add_argument('-s', '--start', default=None, type=int, help='start idx to process')
    parser.add_argument('-e', '--end', default=None, type=int, help='end idx to process')
    parser.add_argument('-fp', '--fps', default=24, type=int, help='fps output of frames')

    args = parser.parse_args()

    video_ids = args.video_ids
    raw_path = args.raw
    meta_path = args.meta_path
    frame_path = args.frame_path
    start = args.start
    end = args.end
    fps = args.fps

    main(video_ids, raw_path, meta_path, frame_path, start, end, fps)


'''


python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/clipped_videos/432 \
--meta-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/432 \
--frame-path /vision2/u/enguyen/mini_cba/clipped_frames/432 \
--start None \
--end None

python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/clipped_videos/432_redo2 \
--meta-path /scr-ssd/enguyen/normal_1.0x/frames_fps16/meta \
--frame-path /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--fps 16



# running on slowed frames

python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/slowed_clip_videos \
--meta-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/slowed \
--frame-path /vision2/u/enguyen/mini_cba/slowed_clip_videos/frames



python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/slowed_clip_videos \
--meta-path /scr-ssd/enguyen/slowed/432/frames_fps24/meta \
--frame-path /scr-ssd/enguyen/slowed/432/frames_fps24
 


python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/slowed_clips_0.2x \
--meta-path /scr-ssd/enguyen/slowed_0.2x/frames_fps24/meta \
--frame-path /scr-ssd/enguyen/slowed_0.2x/frames_fps24 \
--fps 24


python video_to_frame_and_meta.py \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--raw /vision2/u/enguyen/mini_cba/slowed_clips_0.2x \
--meta-path /scr-ssd/enguyen/slowed_0.2x/frames_fps16/meta \
--frame-path /scr-ssd/enguyen/slowed_0.2x/frames_fps16 \
--fps 16



'''