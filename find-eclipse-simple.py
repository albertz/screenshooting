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
from functools import *


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
	newim = cv.CreateImage((int(w),int(h)), cv.IPL_DEPTH_8U, im.channels)
	interpol = cv.CV_INTER_CUBIC
	#interpol = CV_INTER_LINEAR
	#interpol = CV_INTER_AREA
	cv.Resize(im, newim, interpol)
	return newim

def scaledImage(im, scale):
	w,h = cv.GetSize(im)
	return resizedImage(im, w * scale, h * scale)

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
			c1 = cv.Get2D(im1, y, x)
			c2 = cv.Get2D(im2, y, x)
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
		
def rectCenter(r):
	x1,y1,x2,y2 = r
	return ((x1+x2)/2, (y1+y2)/2)

def dist(v1,v2):
	return sqrt(sum( (a - b) ** 2 for (a,b) in zip(v1,v2) ))

def rectDist(r1,r2):
	return dist(rectCenter(r1), rectCenter(r2))


W = 4
H = 4
eclipseFingerPrint = subImageScaled(eclipseIcon, middleRect(subSquareRect(eclipseFullRect), 0.4), W, H)


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



HaveWindow = False

def showImageWithRects(im, rects = []):
	def draw_rects(im, rects, color = cv.RGB(0,255,0)):
		for x1,y1,x2,y2 in rects:
			cv.Rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)),
						 color, 3, 8, 0)

	imcopy = cv.CloneImage(im)
	draw_rects(imcopy, rects)
	global HaveWindow
	if not HaveWindow:
		cv.NamedWindow("Screenshot", cv.CV_WINDOW_AUTOSIZE)
		HaveWindow = True
	cv.ShowImage("Screenshot", resizedImage(imcopy, 800, 600))
	cv.WaitKey(1)


if False:
	showImageWithRects(resizedImage(eclipseFingerPrint, 100, 100))
	cv.WaitKey(0)
	quit()


def vecMult(r, f):
	return map(lambda x: x * f, r)

def vecSum(r1, r2):
	return map(lambda (x1,x2): x1 + x2, zip(r1,r2))

def scaleRectToSize(r, size):
	return middleRect(r, size / (r[2]-r[0]))


def rectsFromMatchSpots(matches):
	rects = []
	def putIntoRects(newr, rects):
		for i in xrange(len(rects)):
			r,n = rects[i]
			if rectDist(r,newr) < 20:
				r = scaleRectToSize(r, newr[2] - newr[0])
				sizeScale = max(1.0, rectDist(r,newr) / (r[2] - r[0]))
				r = vecSum(vecMult(r, n), newr)
				n += 1
				r = vecMult(r, 1.0/n)
				r = middleRect(r, sizeScale)
				rects[i] = (r,n)
				return
		rects += [(newr,1)]
		
	for r in matches:
		putIntoRects(r, rects)

	return map(itemgetter(0), [(r,n) for (r,n) in rects if n > 3])


class ColorSet:
	def __init__(self, colorNum):
		self.colors = [((0,0,0),0)] * colorNum
	
	def firstUnsetIndex(self):
		for i,(color,n) in izip(xrange(len(self.colors)),self.colors):
			if n == 0: return i
		return len(self.colors)
		
	def bestMatch(self, color):
		endIndex = self.firstUnsetIndex()
		if endIndex == 0: return 0,0
		dists = imap(partial(dist, color), imap(itemgetter(0), self.colors))
		d,i = min(izip(dists,xrange(endIndex)))
		if d > 0 and endIndex < len(self.colors): return endIndex,d
		return i,d
		
	def merge(self, color):
		i,_ = self.bestMatch(color)
		oldcolor,n = self.colors[i]
		color = vecSum(vecMult(oldcolor, n), color)
		n += 1
		color = vecMult(color, 1.0/n)
		self.colors[i] = (color,n)
	
	def distance(self, color):
		_,d = self.bestMatch(color)
		return d

def iteratePosInRect(rect):
	x1,y1,x2,y2 = rect
	for _x in xrange(int(x1),int(x2)):
		for _y in xrange(int(y1),int(y2)):
			yield (_x,_y)

MaxW = 60
MaxH = 60
MaxColorDist = 30

def objectBoundingRect(im, rect):
	colors = ColorSet(10)
	for x,y in iteratePosInRect((0,0,W,H)):
		colors.merge(cv.Get2D(eclipseFingerPrint, y, x)[0:3])
	
	rect = map(int, rect)
	dirs = set()
	x1,y1,x2,y2 = rect
	if x1 > 0: dirs.add(0)
	if y1 > 0: dirs.add(1)
	if x2 < im.width - 1: dirs.add(2)
	if y2 < im.height - 1: dirs.add(3)
	dir = dirs.__iter__().next()
	while True:			
		if dir >= 2: rect[dir] += 1
		else: rect[dir] -= 1
		x1,y1,x2,y2 = rect

		if x2-x1 > MaxW: break
		if y2-y1 > MaxH: break

		if x1 <= 0: dirs.remove(0)
		if y1 <= 0: dirs.remove(1)
		if x2 >= im.width: dirs.remove(2)
		if y2 >= im.height: dirs.remove(3)
		
		if dir == 0: newpositions = [(x1,_y) for _y in xrange(y1,y2)]
		if dir == 1: newpositions = [(_x,y1) for _x in xrange(x1,x2)]
		if dir == 2: newpositions = [(x2-1,_y) for _y in xrange(y1,y2)]
		if dir == 3: newpositions = [(_x,y2-1) for _x in xrange(x1,x2)]
		
		pixelcolors = [cv.Get2D(im, y,x)[0:3] for (x,y) in newpositions]
		colordists = [colors.distance(c) for c in pixelcolors]
		#print dir, min(izip(colordists,newpositions))
		if min(colordists) < MaxColorDist:
			for color,colordist in izip(pixelcolors,colordists):
				if colordist < MaxColorDist:
					colors.merge(color)
		else: # too much color diff -> stop in this direction
			dirs.remove(dir)
		
		if len(dirs) == 0: break
		while True:
			dir = (dir + 1) % 4
			if dir in dirs: break
	
	return tuple(rect)

MinW = 15
MinH = 15
MaxWHDiff = 1.5

def rectIconConditionsAreOk(rect):
	x1,y1,x2,y2 = rect
	w = x2-x1
	h = y2-y1
	if w > MaxW or w < MinW: return False
	if h > MaxH or h < MinH: return False
	if w > h: f = float(w) / h
	else: f = float(h) / w
	if f > MaxWHDiff: return False
	return True
	

def checkFile(f):
	sys.stdout.write(f + " :")
	sys.stdout.flush()
	im = cv.LoadImage(f)

	matches = bestNMatches(im)	
	matches = rectsFromMatchSpots(matches)
	
	matches = map(partial(objectBoundingRect, im), matches)
	matches = [r for r in matches if rectIconConditionsAreOk(r)]
	
	print zip(matches, [ (r[2] - r[0] + r[3] - r[1]) / 2 for r in matches ])
	
	#matches = map(partial(middleRect, scale = 2.5), matches)
	
	showImageWithRects(im, matches)
	if cv.WaitKey(0) == ord('q'): quit()
	#cv.DestroyWindow("Screenshot")
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
		files = glob("2010-10-*.png")
		#files = glob("2010-10-11.*.png") # bottom dock with eclipse
		#files = glob("2010-10-28.*.png") # left dock with eclipse
		random.shuffle(files)
		
		# some problematic ones
		files = ["2010-10-16.23.59.32.png"] + files
		
	else:
		files = sys.argv[1:]

	for f in files: checkFile(f)

