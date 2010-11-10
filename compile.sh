#!/bin/sh
g++ find-eclipse.cpp -o find-eclipse -I /usr/local/include/opencv -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_features2d -lopencv_imgproc -lopencv_calib3d

