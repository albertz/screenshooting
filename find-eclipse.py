#!/usr/bin/python -u -O

from glob import glob
import random
import cv
import sys
from math import *
dock = __import__("find-dock")




# we need this hack because there is a memleak in cv.CreateHist

def __create_default_hist__pool_entry():
	range = [0, 255]
	ranges = [range, range, range]
	hist = cv.CreateHist([10,10,10], cv.CV_HIST_ARRAY, ranges, 1)
	return hist

# 2 entries should be enough for all our use cases
__hist_pool = [ __create_default_hist__pool_entry() for i in xrange(2) ]
__hist_pool_index = 0

def __create_default_hist():
	global __hist_pool, __hist_pool_index
	hist = __hist_pool[__hist_pool_index]
	__hist_pool_index += 1
	__hist_pool_index %= len(__hist_pool)
	cv.ClearHist(hist)
	return hist
	
	
	
def __hs_histogram_base(src, hist):
	srcsize = cv.GetSize(src)
	
	r_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	g_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	b_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	cv.Split(src, r_plane, g_plane, b_plane, None)
	planes = [r_plane, g_plane, b_plane]
	#drawImage(r_plane)
	
	cv.CalcHist([cv.GetImage(i) for i in planes], hist, 1)


def hist_for_im_rects(im, rects):
	hist = __create_default_hist()
	oldROI = cv.GetImageROI(im)
	for x1,y1,x2,y2 in rects:
		rect = ( max(int(x1),0), max(int(y1),0), min(int(x2),im.width), min(int(y2),im.height) )
		if rect[0] >= rect[2] or rect[1] >= rect[3]: continue
		rect = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
		cv.SetImageROI(im, rect)
		__hs_histogram_base(im, hist)
	cv.SetImageROI(im, oldROI)
	cv.NormalizeHist(hist, 1.0)
	return hist

def hist_for_im_rect(im, rect):
	return hist_for_im_rects(im, [rect])

def compareColorsInAreas(im1, rects1, im2, rects2):
	hist1 = hist_for_im_rects(im1, rects1)
	hist2 = hist_for_im_rects(im2, rects2)
	#method = cv.CV_COMP_CHISQR
	method = cv.CV_COMP_BHATTACHARYYA
	histDiff = cv.CompareHist(hist1, hist2, method)
	
	#histDiff = 1 - histDiff
	#print histDiff
	return histDiff
	

def subarea(rect, ix, iy, n):
	n = float(n)
	x1,y1,x2,y2 = rect
	w = x2 - x1
	h = y2 - y1
	w /= n
	h /= n
	x1,y1 = x1 + ix*w, y1 + iy*h
	x2,y2 = x1 + w, y2 + h
	return (x1,y1,x2,y2)

def resizedImage(im, w, h):
	newim = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, im.channels)
	interpol = cv.CV_INTER_CUBIC
	#interpol = CV_INTER_LINEAR
	#interpol = CV_INTER_AREA
	cv.Resize(im, newim, interpol)
	return newim

SubImageCache = dict()

def subImageScaled(im, rect, w, h):
	if im == eclipseIcon:
		cacheKey = (rect, w, h)
		if cacheKey in SubImageCache: return SubImageCache[cacheKey]
	else:
		cacheKey = None
	rect = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
	cv.SetImageROI(im, rect)
	resizedim = resizedImage(im, w, h)
	if cacheKey: SubImageCache[cacheKey] = resizedim
	return resizedim
	
def compareAreas(im1, rect1, im2, rect2):
	n = 10
	c = 0
	im1 = subImageScaled(im1, rect1, n, n)
	im2 = subImageScaled(im2, rect2, n, n)
	
	values = []
	for x in range(n):
		for y in range(n):
			c1 = cv.Get2D(im1, x, y)
			c2 = cv.Get2D(im2, x, y)
			#print x,",",y,":",c1,c2
			if c2[0:3] == (255,255,255): continue
			#weight = sum( (255 - c2[i]) / 255.0 for i in [0,1,2] ) / 3
			values += [ sqrt( sum( [ (abs(c1[i] - c2[i]) / 255.0) ** 2 for i in [0,1,2] ] ) ) ]
	#print values
	return sqrt(sum([ x*x for x in values ]))
	return sum(values) / len(values)

def compareAreasVariable(im1, rect1, im2):
	step = 10
	dx1 = 1
	dx2 = 2
	dy1 = 1
	dy2 = 2
	x1,y1 = dx1*step, dy1*step
	x2,y2 = im2.width - x1, im2.height - y1
	while step > 0:
		values = []
		for _x1 in xrange(max(x1 - dx1*step, 0), x1 + dx1*step+1, step):
			for _y1 in xrange(max(y1 - dy1*step, 0), y1 + dy1*step+1, step):
				for _x2 in xrange(x2 - dx2*step, min(x2 + dx2*step, im2.width)+1, step):
					for _y2 in xrange(y2 - dy2*step, min(y2 + dy2*step,im2.height)+1, step):
						values += [ (compareAreas(im1, rect1, im2, (_x1,_y1,_x2,_y2)), (_x1,_y1,_x2,_y2)) ]
		value, (x1,y1,x2,y2) = min(values)
		step -= 2
	return value, (x1,y1,x2,y2)
	
def diffImage(im1, rect1, im2, rect2):
	w,h = 20,20
	im1 = subImageScaled(im1, rect1, w, h)
	im2 = subImageScaled(im2, rect2, w, h)
	newim = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, 3)
	for x in xrange(w):
		for y in xrange(h):
			c1 = cv.Get2D(im1, x, y)[0:3]
			c2 = cv.Get2D(im2, x, y)[0:3]
			if c2[0:3] == (255,255,255): newc = (0,0,0)
			else: newc = tuple( abs(c1[i] - c2[i]) for i in [0,1,2] )
			cv.Set2D(newim, x, y, newc)
	return newim

if len(sys.argv) <= 1:
	#files = glob("*.png")
	#files = glob("2010-10-*.png")
	files = glob("2010-10-11.*.png") # bottom dock with eclipse
	#files = glob("2010-10-28.*.png") # left dock with eclipse
	random.shuffle(files)
else:
	files = sys.argv[1:]

def showImage(im, rect = None, wait = True, window = "icon"):
	if rect:
		rect = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
		cv.SetImageROI(im, rect)
		imcopy = cv.CloneImage(im)
	else:
		imcopy = im
	cv.ShowImage(window, imcopy)
	cv.SetImageROI(im, (0,0,im.width,im.height))
	if wait:
		key = cv.WaitKey(0)
	else:
		key = cv.WaitKey(1)
	if key == ord('q'): quit()
		
eclipseIcon = cv.LoadImage("eclipse-icon.png", cv.CV_LOAD_IMAGE_UNCHANGED)
eclipseFullRect = (0,0,eclipseIcon.width,eclipseIcon.height)

for f in files:
	sys.stdout.write(f + " :")
	sys.stdout.flush()
	im = cv.LoadImage(f)
	sys.stdout.write(".")
	sys.stdout.flush()

	#showImage(im, wait = False, window = "Screenshot")
	iconrects = dock.getDockIcons(im, allSides = True)
	sys.stdout.write("*")
	sys.stdout.flush()

	if len(iconrects) == 0:
		print ": no eclipse (no icons at all)"
		continue
	
	iconprobs = []
	for iconrect in iconrects:
		prob,eclipseRect = compareAreasVariable(im, iconrect, eclipseIcon)
		sys.stdout.write(".")
		sys.stdout.flush()
		iconprobs += [(prob,iconrect,eclipseRect)]
		#print "  ~:", prob, eclipseRect
		#showImage(subImageScaled(im, iconrect, 200, 200), wait = False)
		#showImage(resizedImage(diffImage(im, iconrect, eclipseIcon, eclipseRect), 200, 200), window = "diff")

	iconprobmin,iconrect,eclipseRect = min(iconprobs)
	if iconprobmin < 2: # 2 seems to be good :p (1.7634 mostly for eclipse)
		print ": found with", iconprobmin, iconrect, eclipseRect
		showImage(subImageScaled(im, iconrect, 200, 200), window = "diff")
	else:
		print ": no eclipse (min is", iconprobmin, "at", iconrect, ")"
	