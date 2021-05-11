import ffmpeg
import json
import os


clip_ids_path = '/vision2/u/enguyen/cpr-detection/post_processing_code/data/432/clip_filenames.json'

with open(clip_ids_path) as f:
    clip_ids_list = json.load(f)['clip_filenames']


path_to_clips = '/vision2/u/enguyen/mini_cba/clipped_videos/432_redo2'
out_path = '/vision2/u/enguyen/mini_cba/clipped_videos/432_redo2/slowed2'

os.makedirs(out_path, exist_ok=True)

failed = []

for clip_id in clip_ids_list:

    # import pdb; pdb.set_trace()

    clip_file = clip_id + '.mp4'
    clip_input_path = os.path.join(path_to_clips, clip_file)
    clip_output_path = os.path.join(out_path, clip_file)

    try:
        # use python wrapper for ffmpeg (more stable than ffmpeg in terminal)
        (ffmpeg.input(clip_input_path)
            .setpts('2*PTS')
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



