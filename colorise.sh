#/bin/bash

# Separate frames
mkdir -p frames
ffmpeg -i $1 frames/$filename%03d.jpg
# Colorise frames
for f in frames/*; do python batch.py $f; done
# Extract audio
ffmpeg -i $1 -vn -acodec copy audio.aac
# Stitch frames
ffmpeg -f image2 -i casa/%03d_c.jpg -r 12 ${1}_c.avi
# Apply audio
ffmpeg -i ${1}_c.avi -i audio.aac -codec copy -shortest output.avi
