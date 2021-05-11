#!/bin/bash


# run this in the directory of the .mp4 videos, and change the output dir

for clip_name in *.mp4; do
    ffmpeg -i $clip_name -vf "setpts=2*PTS" -an slowed/$clip_name
done