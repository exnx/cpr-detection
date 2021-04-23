# import imageio
import cv2
import argparse
import json
import os



def main(video_id_path, input_path, output_path):

    with open(video_id_path) as f:
        video_ids = json.load(f).keys()

    fail_ids = []
    success_id = {}

    for video_id in video_ids:

        video_path = os.path.join(input_path, video_id) + ".mp4"

        try:
            frame_dir = os.path.join(output_path, video_id)
            os.makedirs(frame_dir)

            vidcap = cv2.VideoCapture(video_path)

            success, img = vidcap.read()
            count = 0

            while success:

                frame_name = str(count).zfill(5) + ".jpg"
                frame_path = os.path.join(frame_dir, frame_name)

                # print('frame_path', frame_path)

                cv2.imwrite(frame_path, img)  # save frame as JPEG file
                success, img = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1

            # track frame count
            success_id[video_id] = count
            print('successful - id: {}, count:{}'.format(video_id, count))

        except Exception as e:
            # need to log video id to
            print(e)
            print('failed id', video_id)
            fail_ids.append(video_id)


    for f_id in fail_ids:
        print('failed id:', f_id)


        # reader = imageio.get_reader('imageio:cockatoo.mp4')
        #
        # for frame_number, im in enumerate(reader):
        #     # im is numpy array
        #     if frame_number % 10 == 0:
        #         imageio.imwrite(f'frame_{frame_number}.jpg', im)



if __name__ == '__main__':

    print('started...')

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video-path', help='path to video id json')
    parser.add_argument('-i', '--input-path', help='path to input mp4 videos')
    parser.add_argument('-o', '--output-path', default=None, help='path to spit out frames')

    args = parser.parse_args()

    video_id_path = args.video_path
    input_path = args.input_path
    output_path = args.output_path

    main(video_id_path, input_path, output_path)


'''

python video_to_frames.py \
    --video-path missing_list.json \
    --input-path /vision/group/miniCBA/download_video/videos/allVideos \
    --output-path /vision2/u/enguyen/mini_cba/test_fps10/

'''