# importing cv2
import cv2
import argparse
import json
import os
import pandas as pd


base_rate = 111.9

def read_csv_as_df(path):
    df = pd.read_csv(path)
    return df


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


def write_on_frames(frames, output, outputs_avg, rate_label):

    '''

    :param frames: list of frames
    :param output: float, is the speed factor
    :param rate_label: float, ground truth rate
    :return:
    '''


    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font

    output_loc = (50, 50)
    rate_pred_loc = (50, 75)
    rate_avg_loc = (50, 100)
    rate_label_loc = (50, 125)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    # blue = (255, 0, 0)
    green = (0, 255, 0)
    # red = (0, 0, 255)

    # ranges
    # 0.0 - 0.3 is red
    # 0.3 - 0.49 is blue
    # 0.50 - 1.00 is green


    frames_with_text = []

    # import pdb; pdb.set_trace()

    output_text = '{:.2f}x'.format(output)
    rate_pred_text = 'pred: {:.1f}'.format(int(base_rate*output))
    rate_avg_text = 'avg pred: {:.1f}'.format(int(base_rate*outputs_avg))
    rate_label_text = 'avg label: {:.1f}'.format(rate_label)

    font_color = green

    for frame in frames:
        resized = cv2.resize(frame, (width, height))

        # Using cv2.putText() method
        cv2.putText(resized, output_text, output_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_pred_text, rate_pred_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_avg_text, rate_avg_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_label_text, rate_label_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)

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
    # loop thru videos
    # loop thru pred/target
    # get frame list
    # retrieve frames
    # write target/labels on frames
    # save frames to path

    results_dir = args.results_dir
    frame_dir = args.frame_dir
    out_dir = args.out_dir
    rate_labels_dir = args.rate_labels

    results_json = read_json(results_dir)['results']

    # new for using labels
    rate_labels_df = read_csv_as_df(rate_labels_dir)
    rate_video_ids = rate_labels_df['video_id'].tolist()
    rate_labels = rate_labels_df['rate'].tolist()

    total_mae = 0
    count = 0

    # loop thru video
    for i in range(len(rate_video_ids)):

        video_id = rate_video_ids[i]
        rate_label = rate_labels[i]

        print('processing video:', video_id)

        # get model output from from results
        video_json = results_json[video_id]

        out_video_dir = os.path.join(out_dir, video_id)

        # make a new dir
        os.makedirs(out_video_dir, exist_ok=True)

        # preds = video_json['preds']
        outputs = video_json['outputs']

        outputs_avg = 0
        for o in range(len(outputs)):
            outputs_avg += outputs[o][0]

        outputs_avg /= len(outputs)

        segments = video_json['segments']

        # calc prefix
        prefix = os.path.join(frame_dir, video_id)

        # need to track the count so far...

        # loop thru clips
        for j in range(len(outputs)):

            # calc error
            output = outputs[j][0]
            mae = abs(output*base_rate - rate_label)
            total_mae += mae
            count += 1

            # import pdb; pdb.set_trace()

            # retrieve frames
            frames, frame_names = get_frames(prefix, segments[j])

            # write on frames (all with same text)
            frames_with_text = write_on_frames(frames, output, outputs_avg, rate_label)

            save_frames(frames_with_text, out_dir, video_id, frame_names)

    print('MAE: {:.2f}, segments count: {}'.format(total_mae / count, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-dir', help='dir to the inference results file')
    parser.add_argument('-v', '--frame-dir', help='root dir to all frames')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')
    parser.add_argument('-rl', '--rate-labels', default=None, help='path to rate labels')

    args = parser.parse_args()

    main(args)

'''

python write_rate_on_frames_mse.py \
--results-dir /vision2/u/enguyen/results/rate_pred/run8_res18_mse_action_pretrained/inference_results_last_segment/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_res18_mse_action_pretrained_last_segment/frames_corrected \
--rate-labels /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv


'''
















