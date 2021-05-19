import ffmpeg
import json
import os
import argparse



def main(args):

    pts_slow_factor = args.pts_slow_factor
    clip_ids_path = args.clip_ids_path
    path_to_clips = args.path_to_clips
    out_path = args.out_path
    start = args.start
    end = args.end

    with open(clip_ids_path) as f:
        clip_ids_list = json.load(f)['clip_filenames']

    # for processing in chunks
    if start is not None:
        clip_ids_list = clip_ids_list[start:end]

    os.makedirs(out_path, exist_ok=True)

    failed = []

    for clip_id in clip_ids_list:

        clip_file = clip_id + '.mp4'
        clip_input_path = os.path.join(path_to_clips, clip_file)
        clip_output_path = os.path.join(out_path, clip_file)

        try:
            # use python wrapper for ffmpeg (more stable than ffmpeg in terminal)
            (ffmpeg.input(clip_input_path)
                .setpts('{}*PTS'.format(pts_slow_factor))
                .output(clip_output_path)
                .run())

        except:
            print('failed', clip_id)
            failed.append(clip_id)


    print('failed list', failed)

    out_meta = os.path.join(out_path, 'failed.json')

    with open(out_meta, 'w', encoding='utf-8') as f:
        data_dict = {'failed': failed}
        json.dump(data_dict, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pts-slow-factor', help='factor to slow videos down')
    parser.add_argument('-c', '--clip-ids-path', help='path to video ids')
    parser.add_argument('-pc', '--path-to-clips', default=None, help='path to raw clips')
    parser.add_argument('-o', '--out-path', default=None, help='out path for slowed videos, and failed list')
    parser.add_argument('-s', '--start', default=None, type=int, help='start idx to process')
    parser.add_argument('-e', '--end', default=None, type=int, help='end idx to proc„ess')

    args = parser.parse_args()

    main(args)


'''

python slow_videos.py \
--pts-slow-factor 5 \
--clip-ids-path /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json \
--path-to-clips /vision2/u/enguyen/mini_cba/clipped_videos/432_redo2 \
--out-path /vision2/u/enguyen/mini_cba/slowed_clips_0.2x \
--start None \
--end None







'''