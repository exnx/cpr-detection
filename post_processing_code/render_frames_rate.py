import argparse
import json
import os
import pandas as pd
import numpy as np
import cv2



def read_json(path):
    try:
        with open(path) as f:
            data_dict = json.load(f)

    except Exception as e:
        print(e)
        raise Exception

    return data_dict


def save_frames(frames, prefix, frame_names):

    for i in range(len(frames)):
        frame_path = os.path.join(prefix, frame_names[i])

        cv2.imwrite(frame_path, frames[i])

    return


def write_on_frames(frames,
                    prev_rep_count,
                    rate_pred,
                    rate_pred_avg,
                    rate_label,
                    count_label,
                    fps=16,
                    base_rate=109):

    '''

    :param frames: list of frames
    :param output: float, is the speed factor
    :param rate_label: float, ground truth rate
    :return:
    '''

    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font

    rate_label_loc = (25, 50)
    rate_avg_loc = (25, 75)
    rate_pred_loc = (25, 100)
    output_loc = (25, 125)
    count_loc = (25, 150)
    # count_label_loc = (50, 175)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    green = (0, 255, 0)

    frames_with_text = []

    rate_label_text = 'Avg label: {:.1f}'.format(rate_label)
    rate_avg_text = 'Avg pred: {:.1f}'.format(int(rate_pred_avg))
    rate_pred_text = 'Inst. pred: {:.1f}'.format(int(rate_pred))
    output_text = 'output: {:.2f}x'.format(rate_pred/base_rate)

    font_color = green

    for i, frame in enumerate(frames):
        # calculate current count for frame, round down
        window_fraction = (i + 1 / len(frames))
        curr_rep_count = prev_rep_count + rate_pred * window_fraction / fps / 60

        count_text = 'count: {}/{}'.format(int(curr_rep_count), int(count_label))

        resized = cv2.resize(frame, (width, height))

        cv2.putText(resized, output_text, output_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_pred_text, rate_pred_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_avg_text, rate_avg_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_label_text, rate_label_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, count_text, count_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        # cv2.putText(resized, count_label_text, count_label_loc, font,
        #                   fontScale, font_color, thickness, cv2.LINE_AA)

        frames_with_text.append(resized)

    return frames_with_text


def get_frames(prefix, segment):
    ext = ".jpeg"

    images = []
    img_names = []

    for i in range(segment[0], segment[1]):
        name = str(i).zfill(5) + ext
        file_path = os.path.join(prefix, name)

        try:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                img_names.append(name)
            else:
                print('img is not', file_path)

        except Exception as e:
            print('cannot open img:'.format(file_path))
            print(e)

    return images, img_names


def main(args):
    '''
    loop thru videos
        loop thru pred/target
            get frame list
            retrieve frames
            write target/labels on frames
            save frames to path
    '''


    results_dir = args.results_dir
    out_dir = args.out_dir
    fps = args.fps
    window = args.window
    base_rate = args.base_rate

    results_json = read_json(results_dir)['video_results']

    # loop thru video
    for video_id, result_json in results_json.items():

        print('processing video:', video_id)

        prefix = os.path.join(out_dir, video_id)
        os.makedirs(prefix, exist_ok=True)  # make a new dir for each video
        frame_dir = os.path.join(args.frame_dir, video_id)

        count_label = result_json['count_label']
        rate_label = result_json['rate_label']  # for whole clip
        video_rate_pred_avg = result_json['rate_avg_pred']
        per_frame_rate = result_json['per_frame_rate']
        num_frames = len(per_frame_rate)

        curr_rep_count = 0

        # write on frames by window size batches
        for i in range(0, num_frames, window):
            # retrieve frames
            start = i
            end = min(i + window, num_frames)
            frames, frame_names = get_frames(frame_dir, [start, end])
            rate_pred = np.average(per_frame_rate[start:end])

            # write on frames (all with same text)
            frames_with_text = write_on_frames(frames, curr_rep_count, rate_pred,
                                               video_rate_pred_avg, rate_label, count_label, fps, base_rate)

            save_frames(frames_with_text, prefix, frame_names)

            # make sure to do this after writing/saving frames
            curr_rep_count = curr_rep_count + rate_pred * window / fps / 60

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-dir', help='dir to the inference results file')
    parser.add_argument('-v', '--frame-dir', help='root dir to all frames')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')
    parser.add_argument('-f', '--fps', default=16, type=int, help='fps')
    parser.add_argument('-w', '--window', default=24, type=int, help='sliding window size')
    parser.add_argument('-b', '--base-rate', default=109, type=int, help='rate multiplier for the model output')

    args = parser.parse_args()

    main(args)

'''

python render_frames.py \
--results-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt24/frames_level_results/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt24/rendered_frames/




'''

