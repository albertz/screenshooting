#!/usr/bin/python -u

from glob import glob
import random
import cv
import sys
import heapq
from math import *
from itertools import *
from operator import *
from array import *


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
	if cv.GetSize(im) == (w,h): return im
	rect = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
	rect = tuple(map(int, rect))
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


def middleRect(r, scale = 0.5):
	x1,y1,x2,y2 = r
	w = x2 - x1
	h = y2 - y1
	scale = (1.0 - scale) / 2.0
	x1 += w * scale
	x2 -= w * scale
	y1 += h * scale
	y2 -= h * scale
	return (x1,y1,x2,y2)

def subSquareRect(r):
	x1,y1,x2,y2 = r
	w = x2 - x1
	h = y2 - y1
	if w < h:
		return (x1, y1 + (h - w)/2, x2, y2 - (h - w)/2)
	return (x1 + (w - h)/2, y1, x2 - (w - h)/2, y2)
		

W = 4
H = 4
eclipseFingerPrint = subImageScaled(eclipseIcon, middleRect(subSquareRect(eclipseFullRect), 0.3), W, H)


def bestNMatches__manuel(im, n = 100):
	DockSize = 200
	def subRects(rect):
		x1,y1,x2,y2 = rect
		for _x1 in xrange(x1, x2 - W, W/2):
			for _y1 in xrange(y1, y2 - H, H/2):
				_x2 = _x1 + W
				_y2 = _y1 + H
				yield (_x1,_y1,_x2,_y2)
	leftRect = (0,0,DockSize,im.height)
	rightRect = (im.width-DockSize,0,im.width,im.height)
	bottomRect = (DockSize,im.height-DockSize,im.width-DockSize,im.height)
	allSubRects = chain( subRects(leftRect), subRects(rightRect), subRects(bottomRect) )

	def match2(subrect, dx, dy):
		values = []
		for x in range(W):
			for y in range(H):
				c1 = cv.Get2D(im, subrect[0] + x, subrect[1] + y)
				c2 = cv.Get2D(eclipseFingerPrint, (x + dx) % W, (y + dy) % H)
				values += [ sqrt( sum( [ (abs(c1[i] - c2[i]) / 255.0) ** 2 for i in [0,1,2] ] ) ) ]
		return sqrt(sum([ x*x for x in values ]))

	def match(subrect):
		#sys.stdout.write(".")
		#sys.stdout.flush()
		return min(match2(subrect,dx,dy) for dx in range(W) for dy in range(H))

	def matchRects(rects):
		for r in rects:
			yield (match(r),r)
		
	nlargest = heapq.nlargest(n, matchRects(allSubRects))
	return map(itemgetter(1), nlargest)

def bestNMatches(im, n = 100):
	matchImage = cv.CreateImage((im.width-W+1,im.height-H+1), cv.IPL_DEPTH_32F, 1)
	cv.MatchTemplate(im, eclipseFingerPrint, matchImage, cv.CV_TM_SQDIFF)

	matches = array('f')
	matches.fromstring(matchImage.tostring())
	nlargest = heapq.nsmallest(n, zip(matches, xrange(matchImage.width * matchImage.height)))
	rawPositions = map(itemgetter(1), nlargest)
	positions = map(lambda i: (i % matchImage.width, i / matchImage.width), rawPositions)
	rects = map(lambda (x,y): (x,y,x+W,y+H), positions)
	return rects



def showImageWithRects(im, rects = []):
	def draw_rects(im, rects, color = cv.RGB(0,255,0)):
		for x1,y1,x2,y2 in rects:
			cv.Rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)),
						 color, 3, 8, 0)

	imcopy = cv.CloneImage(im)
	draw_rects(imcopy, rects)
	#cv.NamedWindow('Screenshot', cv.CV_WINDOW_AUTOSIZE)
	cv.ShowImage("Screenshot", imcopy)
	cv.WaitKey(1)


if False:
	showImageWithRects(resizedImage(eclipseFingerPrint, 100, 100))
	cv.WaitKey(0)
	quit()

def checkFile(f):
	sys.stdout.write(f + " :")
	sys.stdout.flush()
	im = cv.LoadImage(f)

	matches = bestNMatches(im)
	#print matches
	
	showImageWithRects(im, matches)	
	if cv.WaitKey(0) == ord('q'): quit()
	cv.DestroyAllWindows()
	return
	
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
	

if __name__ == "__main__":
	if len(sys.argv) <= 1:
		#files = glob("*.png")
		#files = glob("2010-10-*.png")
		files = glob("2010-10-11.*.png") # bottom dock with eclipse
		#files = glob("2010-10-28.*.png") # left dock with eclipse
		random.shuffle(files)
	else:
		files = sys.argv[1:]

	for f in files: checkFile(f)

