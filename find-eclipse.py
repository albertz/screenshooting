#!/usr/bin/python

import cv
from glob import glob
cs = cv.Load("eclipse-icon.xml")

for f in glob("2010-10-11.*.png"):
	im = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
	print f, ":", cv.HaarDetectObjects(im, cs, cv.CreateMemStorage(0), min_neighbors=1, min_size=(10,10))
