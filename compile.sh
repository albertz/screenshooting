#!/bin/zsh
for f in *.cpp; do
bin=${f/.cpp/.bin}
echo "$bin .."
g++ $f -o $bin -I /usr/local/include/opencv -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_features2d -lopencv_imgproc -lopencv_calib3d
done

