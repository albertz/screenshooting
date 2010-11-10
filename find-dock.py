#!/usr/bin/python

import cv
from glob import glob
from math import *

cv.NamedWindow('Screenshot', cv.CV_WINDOW_AUTOSIZE)

def draw_rects(im, rects):
	for x,y,w,h in rects:
		cv.Rectangle(im, (x,y), (x+w,y+h),
					 cv.RGB(0, 255, 0), 3, 8, 0)

def random_colors_in_aray(im, rect):
	pass

def hist_image(hist):
	(_, max_value, _, _) = cv.GetMinMaxHistValue(hist)
	h_bins = 40
	s_bins = 40
	scale = 10
	hist_img = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 3)

	for h in range(h_bins):
		for s in range(s_bins):
			bin_val = cv.QueryHistValue_2D(hist, h, s)
			intensity = pow(bin_val, 0.1) * 255
			cv.Rectangle(hist_img,
						 (h*scale, s*scale),
						 ((h+1)*scale - 1, (s+1)*scale - 1),
						 cv.CV_RGB(intensity, intensity, intensity), 
						 cv.CV_FILLED)
	return hist_img
	
def hs_histogram(src):
	# Convert to HSV
	hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
	cv.CvtColor(src, hsv, cv.CV_RGB2HSV)

	# Extract the H and S planes
	h_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
	s_plane = cv.CreateMat(src.rows, src.cols, cv.CV_8UC1)
	cv.Split(hsv, h_plane, s_plane, None, None)
	planes = [h_plane, s_plane]
	
	h_bins = 40
	s_bins = 40
	#h_bins = 30
	#s_bins = 32
	hist_size = [h_bins, s_bins]
	# hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
	h_ranges = [0, 180]
	# saturation varies from 0 (black-gray-white) to
	# 255 (pure spectrum color)
	s_ranges = [0, 255]
	ranges = [h_ranges, s_ranges]
	hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
	cv.CalcHist([cv.GetImage(i) for i in planes], hist)
	cv.NormalizeHist(hist, 1.0)
	return hist


files = glob("2010-10-11.*.png")
i = 0

base_hist = hs_histogram(cv.LoadImageM(files[0]))

while True:
	f = files[i]
	print f
	im = cv.LoadImageM(f)

	cv.ShowImage('Screenshot', im)
	
	hist = hs_histogram(im)
	cv.NamedWindow("H-S Histogram", 1)
	cv.ShowImage("H-S Histogram", hist_image(hist))

	dhist = cv.CompareHist(base_hist, hist, cv.CV_COMP_BHATTACHARYYA)
	print "history diff:", dhist
	
	key = cv.WaitKey(0)
	if key in [27, ord('q')]: quit()
	elif key == 63235: i = min(i + 1, len(files))
	elif key == 63234: i = max(i - 1, 0)
	else: print "key", key, "unknown"
	