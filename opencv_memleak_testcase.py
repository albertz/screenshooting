#!/usr/bin/python

import cv, gc

while True:
	hist = cv.CreateHist([40], cv.CV_HIST_ARRAY, [[0,255]], 1)
	del hist
