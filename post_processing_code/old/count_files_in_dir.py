import os
import argparse
import json
import copy



def main(input_dir, meta_path=None):

    # read meta
    if meta_path is not None:
        with open(meta_path) as f:
            metadata_json = json.load(f)

    count = {}

    zero_files = []

    # loop thru each subdir
        # count files in subdir

    print('input dir', input_dir)

    iter = 0

    for root, subdirectories, files in os.walk(input_dir):
        num_files = len(files)
        print('len of files', num_files)

        base = os.path.basename(root)

        count[base] = len(files)
        if base in metadata_json.keys():
            print('updating new num frames:', base)
            metadata_json[base]['num_new_frames'] = num_files

            if num_files == 0:
                print('zero files found! current count:', len(zero_files))
                zero_files.append(base)

        iter += 1
        if iter == 50:
            break

    dirname = os.path.dirname(meta_path)
    meta_out = os.path.join(dirname, 'new_meta.json')

    # write out json
    if meta_path is not None:
        json_out = copy.deepcopy(metadata_json)
    else:
        json_out = copy.deepcopy(count)

    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=4)


    # if meta_path, then update the new count

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='path to root dir')
    parser.add_argument('-m', '--meta-path', default=None, help='path to metadata to add new frame count')
    args = parser.parse_args()

    input_dir = args.input_dir
    meta_path = args.meta_path

    main(input_dir, meta_path)

'''

python count_files_in_dir.py \
--input-dir /vision2/u/enguyen/mini_cba/compression_fps10 \
--meta-path /vision2/u/enguyen/MOMA_Tools/clip_video_webtool/post_processing_code/video_metadata.json

'''