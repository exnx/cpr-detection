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


def write_on_frames(frames, pred, prob):
    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    pred_loc = (50, 50)
    prob_loc = (50, 75)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    # ranges
    # 0.0 - 0.3 is red
    # 0.3 - 0.49 is blue
    # 0.50 - 1.00 is green

    frames_with_text = []

    prob_text = '{}%'.format(int(prob*100))
    pred_text = 'pred: {}'.format(pred)

    if prob < 0.3:
        font_color = red
    elif prob < 0.5:
        font_color = blue
    else:
        font_color = green

    for frame in frames:
        resized = cv2.resize(frame, (width, height))

        # Using cv2.putText() method
        cv2.putText(resized, prob_text, prob_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, pred_text, pred_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)

        frames_with_text.append(resized)

    return frames_with_text


def get_frames(prefix, segment):
    ext = ".jpeg"

    images = []
    img_names = []

    for i in range(segment[0], segment[1] + 1):
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

    results_json = read_json(results_dir)['results']

    # loop thru video
    for video_id, video_json in results_json.items():

        print('processing video:', video_id)

        out_video_dir = os.path.join(out_dir, video_id)

        # make a new dir
        os.makedirs(out_video_dir, exist_ok=True)

        preds = video_json['preds']
        probs = video_json['outputs']
        segments = video_json['segments']

        # calc prefix
        prefix = os.path.join(frame_dir, video_id)

        # loop thru clips
        for i in range(len(preds)):

            # retrieve frames
            frames, frame_names = get_frames(prefix, segments[i])

            # write on frames (all with same text)
            frames_with_text = write_on_frames(
                frames, preds[i], probs[i][1])

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

python write_rate_on_frames.py \
--results-dir /vision2/u/enguyen/results/run6_res101_inference_ep20/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps16 \
--out-dir /vision2/u/enguyen/demos/rate_pred/run6/frames


'''
















