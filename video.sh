#!/bin/bash
ffmpeg -r 30 -pattern_type glob -i 'images/*.png' -c:v libvpx-vp9 -crf 30 -b:v 0 -y -an progress.webm
#ffmpeg -r 30 -pattern_type glob -i 'sweep_images/*.png' -c:v libvpx-vp9 -crf 30 -b:v 0 -y -an sweep.webm   