import os
import argparse


def main(args):

    root_path = args.root_dir
    video_dir = args.out_dir
    os.makedirs(video_dir, exist_ok=True)

    for video_id in os.listdir(root_path):
        video_prefix = os.path.join(root_path, video_id)
        video_name = video_id + ".mp4"
        video_out_path = os.path.join(video_dir, video_name)

        os.system("ffmpeg -framerate 16 -i {0}/%05d.jpeg -c:v libx264 {1}".format(video_prefix, video_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', help='root to all frames')
    parser.add_argument('-o', '--out-dir', default=None, help='path to the output dir')

    args = parser.parse_args()

    main(args)



'''

python create_video_from_frames.py \
--root-dir /vision2/u/enguyen/demos/rate_pred/run8_res18_mse_action_pretrained/frames \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_res18_mse_action_pretrained/videos


'''