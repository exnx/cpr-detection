import json
import argparse
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ffmpeg

'''

This script will loop thru annotations, check if they're valid, clip and save the videos.


'''


def main(anno_path, path_to_raw_videos, path_to_clipped_videos):

    print('main called')

    with open(anno_path, 'r') as f:
        anno_json = json.loads(f.read())

    # loop thru videos, and thru its annos, and check if anno is valid (not deleted)
    for video_id in anno_json.keys():

        file_path = os.path.join(path_to_raw_videos, "{}.mp4".format(video_id))


        vid = ffmpeg.probe(file_path)
        print(vid['streams'])

        import pdb;
        pdb.set_trace()


        video_json = anno_json[video_id]



        # loop thru each key/id for an annotation
        for key, _ in video_json.items():
            anno = video_json[key]
            if anno['valid'] == True:

                # retrieve for making/saving clip
                video_url = anno['video_id']
                start_time = anno['start_time']
                end_time = anno['end_time']
                clip_fname = anno['clip_fname']

                # orig video file path
                file_path = os.path.join(path_to_raw_videos, "{}.mp4".format(video_url))

                clip_out_path = os.path.join(path_to_clipped_videos, clip_fname)

                # ffmpeg_extract_subclip(file_path, start_time, end_time, targetname=clip_out_path)

                print('saved clip to:', clip_out_path)

            else:
                print('NOT valid annotation')



if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--anno-file', help='path to annotation json file')
    parser.add_argument('-rv', '--raw', help='path to raw videos')
    parser.add_argument('-cv', '--clipped', default=None, help='out path to clipped video')

    args = parser.parse_args()

    anno_path = args.anno_file
    raw_path = args.raw
    clipped_path = args.clipped

    main(anno_path, raw_path, clipped_path)




# python clipper.py -a ~/Desktop/annotation_info.json -cv ~/Desktop/clipped_videos -rv ~/Desktop/raw_videos

# python splat_videos.py -a ~/Desktop/cc_videos/sample_annotations.json -rv ~/Desktop/cc_videos/ -cv ~/Desktop/clipped_videos




