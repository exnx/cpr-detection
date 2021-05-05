# importing cv2
import cv2
import argparse
import json
import os


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


def write_on_frames(frames, pred, target, is_pred_compression_started, is_target_compression_started, send_alert):
    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    pred_loc = (50, 50)
    target_loc = (50, 75)
    alert_loc = (50, 125)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    # false pos = red
    # false neg = red
    # true pos = green
    # true neg = blue
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    frames_with_text = []

    for frame in frames:

        if pred == 1:
            pred_text = 'pred: compression'
        else:
            if is_pred_compression_started:
                pred_text = 'pred: pause'
            else:
                pred_text = 'pred: not started'

        if target == 1:
            target_text = 'target: compression'
        else:
            if is_target_compression_started:
                target_text = 'target: pause'
            else:
                target_text = 'target: not started'

        if pred == target and target == 1:
            font_color = green

        elif pred != target:
            font_color = red
        # true negatives
        else:
            font_color = blue

        resized = cv2.resize(frame, (width, height))

        # Using cv2.putText() method
        cv2.putText(resized, target_text, target_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, pred_text, pred_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)

        if send_alert:
            cv2.putText(resized, "ALERT Pause too long!", alert_loc, font,
                        fontScale, red, thickness, cv2.LINE_AA)


        frames_with_text.append(resized)

    return frames_with_text


def get_frames(prefix, segment):
    ext = ".jpeg"

    images = []
    img_names = []

    for i in range(segment[0], segment[1] + 1):
        name = str(i).zfill(5) + ext
        file_path = os.path.join(prefix, name)

        img = cv2.imread(file_path)
        images.append(img)
        img_names.append(name)

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

    # contains a dict of videos
    # 'video_id':
    # 'targets': []
    # 'preds': []
    # 'segments': []

    # constants
    ALERT_CLIP_COUNT_THRESH = 6

    results_json = read_json(results_dir)['results']

    # loop thru video
    for video_id, video_json in results_json.items():

        out_video_dir = os.path.join(out_dir, video_id)

        # make a new dir
        os.makedirs(out_video_dir, exist_ok=True)

        targets = video_json['targets']
        preds = video_json['preds']
        segments = video_json['segments']

        # tracking certain events for text
        is_pred_compression_started = False
        is_target_compression_started = False
        compression_clip_counter = 0
        paused_clip_counter = 0
        send_alert = False

        # loop thru clips
        for i in range(len(targets)):

            if preds[i] == 1:

                send_alert = False
                is_pred_compression_started = True

                # if enough compressions occured, then reset the paused clip counter
                if compression_clip_counter >= 2:
                    paused_clip_counter = 0

                compression_clip_counter += 1

            # we only track 0's if compression started already
            if preds[i] == 0 and is_pred_compression_started:

                if paused_clip_counter >= ALERT_CLIP_COUNT_THRESH:
                    send_alert = True

                compression_clip_counter = 0
                paused_clip_counter += 1

            # need to track target started
            if targets[i] == 1:
                is_target_compression_started = True

            # calc prefix
            prefix = os.path.join(frame_dir, video_id)

            # retrieve frames
            frames, frame_names = get_frames(prefix, segments[i])

            # write on frames (all with same text)
            frames_with_text = write_on_frames(frames, preds[i], targets[i], is_pred_compression_started, is_target_compression_started, send_alert)

            save_frames(frames_with_text, out_dir, video_id, frame_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-dir', help='dir to the inference results file')
    parser.add_argument('-v', '--frame-dir', help='root dir to all frames')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')

    args = parser.parse_args()

    results_dir = args.results_dir
    frame_dir = args.frame_dir
    out_dir = args.out_dir

    main(args)

'''

python write_text_on_frames.py \
--results-dir /vision2/u/enguyen/results/pretrain1_cont2/val_1/val.json \
--frame-dir /vision2/u/enguyen/mini_cba/new_fps10/ \
--out-dir /vision2/u/enguyen/mini_cba/frames_with_text2/


'''
















