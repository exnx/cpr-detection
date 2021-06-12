import os
import argparse


def main(args):

    root_path = args.root_dir
    video_dir = args.out_dir
    fps = args.fps
    os.makedirs(video_dir, exist_ok=True)

    for video_id in os.listdir(root_path):
        video_prefix = os.path.join(root_path, video_id)
        video_name = video_id + ".mp4"
        video_out_path = os.path.join(video_dir, video_name)

        os.system("ffmpeg -framerate {0} -i {1}/%05d.jpeg -c:v libx264 {2}".format(fps, video_prefix, video_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', help='root to all frames')
    parser.add_argument('-o', '--out-dir', default=None, help='path to the output dir')
    parser.add_argument('-f', '--fps', default=None, help='fps out')

    args = parser.parse_args()

    main(args)



'''

python create_video_from_frames.py \
--root-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt24/rendered_frames \
--out-dir /vision2/u/enguyen/demos/rate_pred/run8_chpt24/videos

python create_video_from_frames.py \
--root-dir /vision2/u/enguyen/demos/rate_pred/repnet_frames/v3 \
--out-dir /vision2/u/enguyen/demos/rate_pred/repnet_frames/v3/videos \
--fps 30



'''