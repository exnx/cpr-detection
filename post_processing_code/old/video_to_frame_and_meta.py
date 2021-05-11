import json
import argparse
import os
import ffmpeg
import copy

'''
this will loop thru data splits and all video ids, get the meta data, convert to frames,
count the number of frames made. Will log any errors (video ids) at any stage.
'''

new_fps = 10
max_height = 360

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

def downsize(height, width):

    '''
    Downsize the the height to a fixed amount (constant),
    and keep the same aspect ratio
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

    print('new height/width: {}/{}'.format(new_height, new_width))

    return new_height, new_width


def main(datasplit_path, path_to_raw_videos, meta_path, frame_path):

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

    os.makedirs(meta_path, exist_ok=True)

    metadata = {}
    missing_list = []
    conversion_fail = []
    count_off = []

    with open(datasplit_path) as f:
        datasplit_json = json.load(f)

    video_metadata = {}

    # loop thru all data splits
    for split in datasplit_json.keys():
        data_split = datasplit_json[split]

        # loop thru ids
        for video_id in data_split:

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

                # downsize (if necessary)
                new_height, new_width = downsize(meta_data['height'], meta_data['width'])

                # use python wrapper for ffmpeg (more stable than ffmpeg in terminal)
                (ffmpeg.input(video_path)
                 .filter('fps', fps=new_fps)
                 .output(os.path.join(frame_dir, "%5d.jpeg"),
                         video_bitrate='5000k',
                         s='{}x{}'.format(str(new_height), str(new_width)),
                         sws_flags='bilinear',
                         start_number=0)
                 .run(capture_stdout=True, capture_stderr=True))

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
                print('count {}, duration {}, diff {}'.format(new_frame_count, float(meta_data['duration']), diff))
                if diff > new_fps:
                    count_off.append(video_id)

                    print('current off count')
                    for i in count_off:
                        print(i)

                video_metadata['num_new_frames'] = new_frame_count
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
    parser.add_argument('-d', '--datasplit', help='path to video ids')
    parser.add_argument('-rv', '--raw', help='path to raw videos')
    parser.add_argument('-m', '--meta-path', default=None, help='path to meta data output')
    parser.add_argument('-f', '--frame-path', default=None, help='path to frames output')

    args = parser.parse_args()

    datasplit = args.datasplit
    raw_path = args.raw
    meta_path = args.meta_path
    frame_path = args.frame_path

    main(datasplit, raw_path, meta_path, frame_path)


'''
python video_to_frame_and_meta.py \
--datasplit /vision2/u/enguyen/MOMA_Tools/clip_video_webtool/post_processing_code/data_split.json \
--raw /vision/group/miniCBA/download_video/videos/allVideos \
--meta-path /vision2/u/enguyen/mini_cba/new_fps10/metadata \
--frame-path /vision2/u/enguyen/mini_cba/new_fps10/
python video_to_frame_and_meta.py \
--datasplit /vision2/u/enguyen/MOMA_Tools/clip_video_webtool/post_processing_code/data_split.json \
--raw /vision2/u/enguyen/videos/ \
--meta-path /vision2/u/enguyen/mini_cba/temp_fps10/metadata \
--frame-path /vision2/u/enguyen/mini_cba/temp_fps10/

'''