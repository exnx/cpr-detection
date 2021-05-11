import json
import argparse
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

'''

This script will loop thru annotations, check if they're valid, clip and save the videos.


'''


def main(anno_path, path_to_raw_videos, path_to_clipped_videos, video_ids_path, meta_out_path):

    os.makedirs(path_to_clipped_videos, exist_ok=True)

    with open(anno_path, 'r') as f:
        anno_json = json.loads(f.read())

    with open(video_ids_path, 'r') as f:
        video_ids_json = json.loads(f.read())

    failed_ids = []
    clip_filenames = []

    clips_by_video_id = {} # track clips by video ids

    # loop thru videos, and thru its annos, and check if anno is valid (not deleted)
    for video_id in anno_json.keys():

        # need to make sure we're using only the ids verified in the id list
        if video_id not in video_ids_json.keys():
            continue

        video_json = anno_json[video_id]  # grab annots
        clips_by_video_id[video_id] = []

        # loop thru each key/id for an annotation
        for anno_id, anno in video_json.items():
            if anno['valid'] == True:

                # retrieve for making/saving clip
                # video_url = anno['video_id']
                start_time = anno['start_time']
                end_time = anno['end_time']
                clip_fname = anno['clip_fname']

                clip_id = clip_fname.split('.')[0]
                orig_video_file = video_id + '.mp4'

                # orig video file path
                orig_video_path = os.path.join(path_to_raw_videos, orig_video_file)
                clip_out_path = os.path.join(path_to_clipped_videos, clip_fname)

                try:
                    ffmpeg_extract_subclip(orig_video_path, start_time, end_time, targetname=clip_out_path)
                    clip_filenames.append(clip_id)
                    clips_by_video_id[video_id].append(clip_id)
                except:
                    print('******** failed ffmpeg id! ********', clip_out_path)
                    failed_ids.append(clip_out_path)
                #
                # print('saved clip to:', clip_out_path)

            else:
                print('###### NOT valid annotation')

    fail_out_path = os.path.join(path_to_clipped_videos, 'failed.json')

    # save to file
    with open(fail_out_path, 'w', encoding='utf-8') as f:
        data_dict = {'failed': failed_ids}
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    clip_filenames_out = os.path.join(path_to_clipped_videos, 'clip_filenames.json')

    # save to file
    with open(clip_filenames_out, 'w', encoding='utf-8') as f:
        data_dict = {'clip_filenames': clip_filenames}
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    clips_by_video_id_path = os.path.join(meta_out_path, 'clips_by_video_id.json')

    # save to file
    with open(clips_by_video_id_path, 'w', encoding='utf-8') as f:
        json.dump(clips_by_video_id, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--anno-file', help='path to annotation json file')
    parser.add_argument('-rv', '--raw', help='path to raw videos')
    parser.add_argument('-cv', '--clipped', help='out path to clipped video')
    parser.add_argument('-vi', '--video-ids', help='video ids to use')
    parser.add_argument('-m', '--meta-out', help='path to metadata list output')

    args = parser.parse_args()

    anno_path = args.anno_file
    raw_path = args.raw
    clipped_path = args.clipped
    video_ids_path = args.video_ids
    meta_out_path = args.meta_out

    main(anno_path, raw_path, clipped_path, video_ids_path, meta_out_path)





'''

python video_clipper.py \
--anno-file /vision2/u/enguyen/cpr-detection/post_processing_code/data/annotation_info.json \
--clipped /vision2/u/enguyen/mini_cba/clipped_videos/432_redo2 \
--raw /vision/group/miniCBA/download_video/videos/allVideos \
--video-ids /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/videoID_verified_labels.json \
--meta-out /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/

'''




