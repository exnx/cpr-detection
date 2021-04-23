import json
import argparse
import os
import ffmpeg
import copy

'''

this will loop thru data splits and all video ids, and get the meta data

'''


def main(datasplit_path, path_to_raw_videos, meta_path):

    metadata = {}
    missing_list = []
    train_data = {}
    train_data['train'] = []

    with open(datasplit_path) as f:
        datasplit_json = json.load(f)

    # loop thru its split types
    for split_type in datasplit_json.keys():

        # list of ids for datasplit type
        data_split = datasplit_json[split_type]

        video_metadata = {}  # create a new video metadata

        for video_id in data_split:

            print('video id', video_id)

            file_path = os.path.join(path_to_raw_videos, "{}.mp4".format(video_id))

            try:
                vid_data = ffmpeg.probe(file_path)['streams'][0]

                # just for getting around missing videos, delete later
                train_data['train'].append(video_id)

            except ffmpeg.Error as e:

                print('video id not found:', video_id)
                missing_list.append(video_id)

                print(e.stderr)

                # print('stdout:', e.stdout.decode('utf8'))
                # raise e

            # import pdb;
            # pdb.set_trace()

            video_metadata['duration'] = vid_data['duration']
            video_metadata['height'] = vid_data['height']
            video_metadata['width'] = vid_data['width']
            video_metadata['num_orig_frames'] = vid_data['nb_frames']
            video_metadata['avg_frame_rate'] = vid_data['avg_frame_rate']

            metadata[video_id] = copy.deepcopy(video_metadata)

    missing_path = os.path.join(meta_path, 'missing_list.txt')
    train_path = os.path.join(meta_path, "train_data.json")
    meta_path = os.path.join(meta_path, "video_metadata.json")



    # with open(meta_path, "w") as f:
    #     f.write(metadata)

    # write metadata to disk
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # for temp testing
    # with open(train_path, 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(missing_path, "w") as f:
        for item in missing_list:
            f.write("%s\n" % item)


if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasplit', help='path to datasplit json file')
    parser.add_argument('-rv', '--raw', help='path to raw videos')
    parser.add_argument('-m', '--meta-path', default=None, help='path to meta data output')

    args = parser.parse_args()

    datasplit_path = args.datasplit
    raw_path = args.raw
    meta_path = args.meta_path

    main(datasplit_path, raw_path, meta_path)


'''

# python clipper.py -a ~/Desktop/annotation_info.json -cv ~/Desktop/clipped_videos -rv ~/Desktop/raw_videos

# python splat_videos.py -a ~/Desktop/cc_videos/sample_annotations.json -rv ~/Desktop/cc_videos/ -cv ~/Desktop/clipped_videos

python create_video_metadata.py \
--datasplit /vision2/u/enguyen/MOMA_Tools/clip_video_webtool/post_processing_code/data_split.json \
--raw /vision/group/miniCBA/download_video/videos/allVideos \
--meta-path /vision2/u/enguyen/MOMA_Tools/clip_video_webtool/post_processing_code/


'''