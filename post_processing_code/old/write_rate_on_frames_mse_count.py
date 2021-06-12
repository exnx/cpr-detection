# importing cv2
import cv2
import argparse
import json
import os
import pandas as pd


base_rate = 111

def read_csv_as_df(path):
    df = pd.read_csv(path)
    return df

def write_json(data_dict, path):

    '''

    :param data_dict: dict to write to json
    :param path: path to json
    :param indent: for easy viewing.  Use None if you want to save a lot of space
    :return:
    '''

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

def read_json(path):
    try:
        with open(path) as f:
            data_dict = json.load(f)

    except Exception as e:
        print(e)
        raise Exception

    return data_dict


def save_frames(frames, out_dir, video_id, frame_names):
    prefix = os.path.join(out_dir, video_id)

    for i in range(len(frames)):
        frame_path = os.path.join(prefix, frame_names[i])

        cv2.imwrite(frame_path, frames[i])

    return


def write_on_frames(frames, prev_rep_count, output, outputs_avg, rate_label, count_label, fps=16):

    '''

    :param frames: list of frames
    :param output: float, is the speed factor
    :param rate_label: float, ground truth rate
    :return:
    '''


    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font

    # output_loc = (50, 50)
    rate_pred_loc = (50, 50)
    rate_avg_loc = (50, 75)
    rate_label_loc = (50, 100)
    count_loc = (50, 125)
    # count_label_loc = (50, 175)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    green = (0, 255, 0)

    frames_with_text = []

    rate_pred = base_rate*output

    # output_text = '{:.2f}x'.format(output)
    rate_pred_text = 'rate: {:.1f}'.format(int(rate_pred))
    rate_avg_text = 'avg pred: {:.1f}'.format(int(base_rate*outputs_avg))
    rate_label_text = 'avg label: {:.1f}'.format(rate_label)
    # count_label_text = 'total count label: {:.1f}'.format(count_label)

    font_color = green

    for i, frame in enumerate(frames):
        # calculate current count for frame, round down
        window_fraction = (i + 1 / len(frames))
        curr_rep_count = prev_rep_count + rate_pred * window_fraction / fps / 60

        count_text = 'count: {}/{}'.format(int(curr_rep_count), int(count_label))

        resized = cv2.resize(frame, (width, height))

        # Using cv2.putText() method
        # cv2.putText(resized, output_text, output_loc, font,
        #                   fontScale, font_color, thickness, cv2.LINE_AA)
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

    fps = args.fps
    window = args.window
    results_dir = args.results_dir
    frame_dir = args.frame_dir
    out_dir = args.out_dir
    rate_labels_dir = args.rate_labels

    results_json = read_json(results_dir)['results']

    # new for using labels
    rate_labels_df = read_csv_as_df(rate_labels_dir)
    rate_video_ids = rate_labels_df['video_id'].tolist()
    rate_labels = rate_labels_df['rate'].tolist()
    count_labels = rate_labels_df['count'].tolist()
    duration_labels = rate_labels_df['duration'].tolist()

    total_count_diff = 0
    total_mae_by_video = 0
    total_mae_by_segment = 0
    segment_count = 0
    curr_video_count = 0

    all_results = {}

    # loop thru video
    for i in range(len(rate_video_ids)):

        video_result = {}

        video_id = rate_video_ids[i]
        print('processing video:', video_id)
        rate_label = rate_labels[i]
        count_label = count_labels[i]
        duration_label = duration_labels[i]
        avg_rate_pred = 0

        if duration_label < 5:
            continue

        curr_video_count += 1
        curr_video_mae = 0

        video_json = results_json[video_id]  # get model output from from results
        out_video_dir = os.path.join(out_dir, video_id)
        os.makedirs(out_video_dir, exist_ok=True)  # make a new dir

        outputs = video_json['outputs']
        num_segments = len(outputs)
        outputs_avg = 0
        for o in range(len(outputs)):
            outputs_avg += outputs[o][0]

        outputs_avg /= num_segments

        segments = video_json['segments']

        # calc prefix
        prefix = os.path.join(frame_dir, video_id)

        # need to track the count so far...
        curr_rep_count = 0

        # loop thru segments
        for j in range(num_segments):
            # calc error
            output = outputs[j][0]
            rate_pred = output*base_rate
            mae = abs(rate_pred - rate_label)
            total_mae_by_segment += mae
            curr_video_mae += mae
            segment_count += 1
            avg_rate_pred += rate_pred

            # if last segment, need to adjust count for partial repeating segment
            if j == len(outputs) - 1:
                # get prev end
                prev_end = segments[j-1][1]
                curr_start = segments[j][0]
                diff = prev_end - curr_start
                count_diff = (diff/window) * (rate_pred * window / fps / 60)
                curr_rep_count -= count_diff
                curr_rep_count = max(curr_rep_count, 0)

            # # retrieve frames
            # frames, frame_names = get_frames(prefix, segments[j])
        #
        #     # write on frames (all with same text)
        #     frames_with_text = write_on_frames(frames, curr_rep_count, output, outputs_avg, rate_label, count_label, fps)
        #
        #     save_frames(frames_with_text, out_dir, video_id, frame_names)
        #
            # make sure to do this after writing/saving frames
            curr_rep_count = curr_rep_count + rate_pred * window / fps / 60

        video_count_diff = abs(curr_rep_count - count_label)
        total_count_diff += video_count_diff
        curr_video_mae /= num_segments  # divide by num segments
        total_mae_by_video += curr_video_mae  # then add for the video

        print('final count for video {}: {}'.format(video_id, int(curr_rep_count)))
        print('MAE for video {}: {:.2f}'.format(video_id, curr_video_mae / num_segments))

        # save video results
        video_result['num_segments'] = num_segments
        video_result['count_pred'] = curr_rep_count
        video_result['count_label'] = count_label
        video_result['count_diff'] = abs(curr_rep_count - count_label)
        video_result['rate_label'] = rate_label
        video_result['rate_avg_pred'] = avg_rate_pred / num_segments
        video_result['rate_mae'] = curr_video_mae

        all_results[video_id] = video_result

    avg_count_diff = total_count_diff / curr_video_count
    mae = total_mae_by_video / curr_video_count
    mae_weighted = total_mae_by_segment / segment_count

    print('\n')
    print('Avg count diff: {:.2f}, video count {}'.format(avg_count_diff, curr_video_count))
    print('MAE: {:.2f}, video count {}'.format(mae, curr_video_count))
    print('MAE (weighted): {:.2f}, segments count: {}'.format(mae_weighted, segment_count))

    results = {'video_results': all_results, 'avg_count_diff': avg_count_diff, 'mae': mae, 'mae_weighted': mae_weighted}

    out_path = os.path.join(out_dir, 'test_results.json')
    write_json(results, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-dir', help='dir to the inference results file')
    parser.add_argument('-v', '--frame-dir', help='root dir to all frames')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')
    parser.add_argument('-rl', '--rate-labels', default=None, help='path to rate labels')
    parser.add_argument('-f', '--fps', default=16, help='fps')
    parser.add_argument('-w', '--window', default=24, help='sliding window size')


    args = parser.parse_args()

    main(args)

'''

python write_rate_on_frames_mse_count.py \
--results-dir /vision2/u/enguyen/results/rate_pred/run8_res18_mse_action_pretrained/inference_results_last_segment/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_res18_mse_action_pretrained_last_segment/frames_fix \
--rate-labels /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv


'''
















