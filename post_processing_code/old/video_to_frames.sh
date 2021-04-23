#!/bin/bash


# valid chest videos json
VALID_VIDS_JSON=$1



# valid chest videos list
#VALID_VIDS_LIST=$1

# location of videos
VIDEO_INPUT_DIR=$2

# output dir for placing frames
OUTPUT_DIR=$3

# Reads from directory itself
# for file_path in $VIDEO_INPUT_DIR/*.mp4; do

#     # get basename of file path
#     dirname=$(basename "${file_path%.*}")
#     dirpath="$OUTPUT_PATH/$dirname"

#     echo $dirpath

#     mkdir -p "$dirpath";

#     # ffmpeg -i "$file_path" -r 10 "$dirpath/%5d.jpeg";

# done






## reads from rows of txt file
#while read video_id; do
#
#  out_path="$OUTPUT_DIR/$video_id"
#
##  mkdir -p $out_path
#
#  file_path="$VIDEO_INPUT_DIR/$video_id.mp4"
#
#  echo $file_path
#
##  # # frame of 10 fps
#  ffmpeg -i "$file_path" -r 10 "$out_path/%5d.jpeg";
#
#done < $VALID_VIDS_LIST
#




# reads from json keys
for video_id in $(jq -r 'values | keys | .[]' $VALID_VIDS_JSON); do

    out_path="$OUTPUT_DIR/$video_id"

    mkdir -p $out_path

    file_path="$VIDEO_INPUT_DIR/$video_id.mp4"

    # # frame of 10 fps
    ffmpeg -i "$file_path" -r 10 "$out_path/%5d.jpeg";

done





# example usage

# local
# sh video_to_frames.sh /Users/ericnguyen/Desktop/MOMA_Tools/clip_video_webtool/post_processing_code/videoID_to_class.json ~/Desktop/cc_videos ~/Desktop/video_frames1

# on cluster
# sh video_to_frames.sh ./missing_list.json /vision/group/miniCBA/download_video/videos/allVideos /vision2/u/enguyen/mini_cba/test_fps10/


