#!/bin/bash

path="respiratory-sound-database/audio_and_txt_files/*.wav"

for f in $path
do
    rep="_16.wav"
    tempFile="${f/.wav/$rep}"
    sox -v 0.99 $f -b 16 -r 8000 $tempFile
    echo "Successfully converted $f to 16 bit depth, 0.99 volume and 8000 sampling rate."
done