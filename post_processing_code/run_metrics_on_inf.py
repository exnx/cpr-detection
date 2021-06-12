# importing cv2
import cv2
import argparse
import json
import os
import pandas as pd
from scipy.signal import medfilt
import numpy as np
from functools import singledispatch
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score



base_rate = 109

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)



@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

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
        json.dump(data_dict, f, ensure_ascii=False, indent=4, default=to_serializable)

def read_json(path):
    try:
        with open(path) as f:
            data_dict = json.load(f)

    except Exception as e:
        print(e)
        raise Exception

    return data_dict


def convert_label(rate_np):
    low_bar = 100
    high_bar = 120

    rate_by_class = np.where(rate_np < low_bar, 0, rate_np)
    rate_by_class = np.where(rate_by_class > high_bar, 2, rate_by_class)
    rate_by_class = np.where((rate_by_class <= high_bar) & (rate_by_class >= low_bar), 1, rate_by_class)

    return rate_by_class


def run_classification(per_frame_rate, rate_label):

    rate_np = np.zeros(len(per_frame_rate))
    rate_np[:] = rate_label

    # constant = np.zeros(len(per_frame_rate))
    # constant[:] = 111.9

    y_test = convert_label(rate_label)
    y_pred = convert_label(per_frame_rate)

    precision, recall, fscore, support = score(y_test, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    return (precision, recall, fscore, support)


def main(args):
    '''
    loop thru videos
        loop thru pred/target
            get frame list
            retrieve frames
            write target/labels on frames
            save frames to path
    '''

    use_smooth = args.use_smooth
    min_duration = args.min_duration
    fps = args.fps
    # window = args.window
    results_dir = args.results_dir
    frame_dir = args.frame_dir
    out_dir = args.out_dir
    rate_labels_dir = args.rate_labels

    os.makedirs(out_dir, exist_ok=True)  # make a new dir

    results_json = read_json(results_dir)['results']

    # new for using labels
    rate_labels_df = read_csv_as_df(rate_labels_dir)
    rate_video_ids = rate_labels_df['video_id'].tolist()
    rate_labels = rate_labels_df['rate'].tolist()
    count_labels = rate_labels_df['count'].tolist()
    duration_labels = rate_labels_df['duration'].tolist()

    total_count_diff = AverageMeter()
    total_mae_by_video = AverageMeter()
    total_rate_diff = AverageMeter()

    curr_video_count = 0

    all_results = {}

    # need to aggregate these for all videos
    all_per_frame_rate = None
    all_rate_label = None

    # loop thru video
    for i in range(len(rate_video_ids)):

        moving_average = StreamingMovingAverage(window_size=11)

        video_result = {}

        video_id = rate_video_ids[i]
        rate_label = rate_labels[i]
        count_label = count_labels[i]
        duration_label = duration_labels[i]
        avg_rate_pred = AverageMeter()

        # for making videos
        # out_video_dir = os.path.join(out_dir, video_id)
        # os.makedirs(out_video_dir, exist_ok=True)  # make a new dir
        #
        if duration_label < min_duration:
            continue

        curr_video_count += 1
        video_json = results_json[video_id]  # get model output from from results
        segments = video_json['segments']

        num_frames = segments[-1][1]  # last end frame

        # need to get per frame rate
        # per_frame_rate_normal = np.zeros(num_frames)
        per_frame_rate = np.zeros(num_frames)
        per_frame_count = np.zeros(num_frames)

        outputs = video_json['outputs']
        num_segments = len(outputs)
        outputs_avg = AverageMeter()
        for o in range(num_segments):
            outputs_avg.update(outputs[o][0])

        # calc prefix
        prefix = os.path.join(frame_dir, video_id)

        last_end = 0

        # loop thru segments
        for j in range(num_segments):
            # calc error
            output = outputs[j][0]
            rate_pred = output*base_rate

            # for trying a constant comparison„
            # rate_pred = 109

            if use_smooth:
                rate_pred = moving_average.process(rate_pred)

            # segment_count += 1
            avg_rate_pred.update(rate_pred)

            start, end = segments[j]

            # if the last segment, then only fill up from the last_end (what is not filled already)
            if j == num_segments - 2:
                start = last_end

            per_frame_rate[start:end] = rate_pred
            per_frame_count[start:end] = rate_pred / (fps * 60)

            last_end = end
            # per_frame_rate_normal[start:end] = rate_pred

        video_rate_avg_pred = np.average(per_frame_rate)
        video_mae = np.average(np.absolute(per_frame_rate-rate_label))
        total_mae_by_video.update(video_mae)  # then add for the video
        video_rate_diff = abs(video_rate_avg_pred-rate_label)
        total_rate_diff.update(video_rate_diff)

        curr_video_rep_count = sum(per_frame_count)
        video_count_diff = abs(curr_video_rep_count - count_label)
        total_count_diff.update(video_count_diff)

        print('{} MAE: {:.2f}, rate diff: {:.2f}'.format(video_id, video_mae, video_rate_diff))

        # save video results
        video_result['per_frame_rate'] = per_frame_rate.tolist()
        video_result['num_segments'] = num_segments
        video_result['count_pred'] = curr_video_rep_count
        video_result['count_label'] = count_label
        video_result['count_diff'] = video_count_diff
        video_result['rate_label'] = rate_label
        video_result['rate_avg_pred'] = video_rate_avg_pred
        video_result['rate_mae'] = video_mae

        all_results[video_id] = video_result

        rate_label_np = np.zeros(len(per_frame_rate))
        rate_label_np[:] = rate_label

        if all_per_frame_rate is None:
            all_per_frame_rate = per_frame_rate
            all_rate_label = rate_label_np
        else:

            try:
                all_per_frame_rate = np.concatenate((all_per_frame_rate, per_frame_rate))
                all_rate_label = np.concatenate((all_rate_label, rate_label_np))
            except:
                import pdb; pdb.set_trace()

    # import pdb;
    # pdb.set_trace()

    # run 3 level classification
    precision, recall, fscore, support = run_classification(all_per_frame_rate, all_rate_label)
    classification_results = {'precision': precision, 'recall': recall, 'fscore': fscore, 'support': support}

    print('\n')
    print('video count:', curr_video_count)
    print('Avg rate diff: {:.2f}'.format(total_rate_diff.avg))
    print('Avg count diff: {:.2f}'.format(total_count_diff.avg))
    print('MAE: {:.2f}'.format(total_mae_by_video.avg))

    results = {'video_results': all_results, 'avg_count_diff': total_count_diff.avg, 'mae': total_mae_by_video.avg, 'classification': classification_results}

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
    parser.add_argument('-m', '--min-duration', type=int, default=5, help='min duration of video to consider')
    parser.add_argument('-u', '--use-smooth', action='store_true', help='whether to smooth or not')

    args = parser.parse_args()

    main(args)

'''

python run_metrics_on_inf.py \
--results-dir /vision2/u/enguyen/results/rate_pred/run8_res18_mse_action_pretrained/inference_chpt24/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt24/frames_level_results \
--rate-labels /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv \
--use-smooth \
--min-duration 0


python run_metrics_on_inf.py \
--results-dir /vision2/u/enguyen/results/rate_pred/run8_res18_mse_action_pretrained/inference_chpt60/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt60/frames_level_results \
--rate-labels /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv \
--use-smooth \
--min-duration 5

 


'''
















