#!/bin/bash

#for dir in */; do
#
#    ffmpeg -framerate 16 -i "${dir///}"/%05d.jpeg -c:v libx264 "${dir///}".mp4
#
#done



# write in a new directory inside called video/

for dir in */; do

    ffmpeg -framerate 16 -i "${dir///}"/%05d.jpeg -c:v libx264 "videos/${dir///}".mp4

done