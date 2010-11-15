#!/usr/bin/python

import cv
from glob import glob
from math import *
from itertools import *
from operator import itemgetter
import random

cv.NamedWindow('Screenshot', cv.CV_WINDOW_AUTOSIZE)

def draw_rects(im, rects, color = cv.RGB(0,255,0)):
	for x1,y1,x2,y2 in rects:
		cv.Rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)),
					 color, 3, 8, 0)

def drawImage(im):
	cv.NamedWindow("Image", cv.CV_WINDOW_AUTOSIZE)
	cv.ShowImage("Image", im)
	if cv.WaitKey(0) == ord('q'): quit()
	cv.DestroyWindow("Image")


def __hs_histogram_base(src, hist):
	srcsize = cv.GetSize(src)
	
	r_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	g_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	b_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	cv.Split(src, r_plane, g_plane, b_plane, None)
	planes = [r_plane, g_plane, b_plane]
	#drawImage(r_plane)
	
	cv.CalcHist([cv.GetImage(i) for i in planes], hist, 1)


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


def hs_histogram(src):
	hist = __create_default_hist()
	__hs_histogram_base(src, hist)
	cv.NormalizeHist(hist, 1.0)
	return hist

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
	#cv.NormalizeHist(hist, 1.0)
	return hist

def hist_for_im_rect(im, rect):
	return hist_for_im_rects(im, [rect])

def compareColorsInAreas(im1, rects1, im2, rects2):
	hist1 = hist_for_im_rects(im1, rects1)
	hist2 = hist_for_im_rects(im2, rects2)
	histDiff = cv.CompareHist(hist1, hist2, cv.CV_COMP_BHATTACHARYYA)
	
	#histDiff = 1 - histDiff
	#print histDiff
	return histDiff
	

def index_filter(ls, *indices):
	return [ ls[i] for i in indices ]

def rectSize(rect):
	x1,y1,x2,y2 = rect
	w = x2 - x1
	h = y2 - y1
	return w * h
	
def rectsSizeSum(rects):
	return sum(rectSize(rect) for rect in rects)


RectProbCache = dict()

class DockRect:
	def __init__(self, im, rect):
		self.im = im
		self.rect = rect
	
	def surrounding_rects(self, space, indices):
		x1,y1,x2,y2 = self.rect
		r1 = (x1 - ((0 in indices) and space or 0), y1, x1, y2) # left
		r2 = (
			x1 - ((0 in indices) and space or 0),
			y1 - ((1 in indices) and space or 0),
			x2 + ((2 in indices) and space or 0),
			y1) # top
		r3 = (x2, y1, x2 + ((2 in indices) and space or 0), y2) # right
		r4 = (
			x1 - ((0 in indices) and space or 0),
			y2,
			x2 + ((2 in indices) and space or 0),
			y2 + ((3 in indices) and space or 0)) # bottom
		return [ r1, r2, r3, r4 ]
	
	def inner_rects(self, indices):
		thick = 1
		x1,y1,x2,y2 = self.rect
		r1 = (x1,y2-thick,x2,y2)
		r2 = (x1,y1,x1+thick,y2)
		r3 = (x2-thick,y1,x2,y2)
		#if x1 == 0: return [r1, r2]
		#return [r1, r3]
		#return [r1]
		return [self.rect]
		
	def probability(self, indices = xrange(0,4), minSize = None, surroundingSpace = None):
		if not minSize: minSize = 30
		innerRects = self.inner_rects(indices)
		if not surroundingSpace: surroundingSpace = 2 #surroundingSpace = max(1, rectsSizeSum(innerRects))
		
		cacheIndex = (tuple(self.rect), tuple(indices), minSize, surroundingSpace)
		global RectProbCache
		if cacheIndex in RectProbCache: return RectProbCache[cacheIndex]
		
		outerRects = self.surrounding_rects(surroundingSpace, indices)
		if rectsSizeSum(innerRects) < minSize: return 0
		if rectsSizeSum(outerRects) < min(surroundingSpace, minSize): return 0
		
		histDiff = compareColorsInAreas(self.im, innerRects, self.im, outerRects)

		RectProbCache[cacheIndex] = histDiff
		return histDiff # the higher the diff, the better the probalitity (we want to have the best sepeaation)


# TODO ... (not used atm)
class IconRectSet:
	def __init__(self, im):
		self.im = im
		

def best_avg_dockrect(dockrects, *probargs):
	rects = [ dockrect.rect for dockrect in dockrects ]
	probs = normProbs([ dockrect.probability(*probargs) for dockrect in dockrects ])	
	probsum = sum(probs)
	rect = [0,0,0,0]
	for dockrect,dockprob in zip(rects,probs):
		for i in [0,1,2,3]:
			rect[i] += dockrect[i] * dockprob / probsum
	return tuple(rect)

def best_dockrect(dockrects, *probargs):
	dockrects = [ (dockrect.rect, dockrect.probability(*probargs)) for dockrect in dockrects ]
	probmax = -100000000
	rect = (0,0,0,0)
	for dockrect,dockprob in dockrects:
		if dockprob >= probmax:
			probmax = dockprob
			rect = dockrect
	#print rect, probmax
	return rect


def iterateRect(x1,y1,x2,y2, maxx, maxy, incindex, maxCount, earlierBreak = 0):
	minx = 0
	miny = 30
	rect = [x1,y1,x2,y2]
	count = 0
	while True:
		x1,y1,x2,y2 = rect
		if x1 < minx or x1 >= maxx: return
		if x2 < minx or x2 > maxx: return
		if y1 < miny or y1 >= maxy: return
		if y2 < miny or y2 > maxy: return
		if incindex == 0 and rect[incindex] < earlierBreak: return
		if incindex == 2 and rect[incindex] >= maxx - earlierBreak: return
		if incindex == 1 and rect[incindex] < earlierBreak: return
		if incindex == 3 and rect[incindex] >= maxy - earlierBreak: return 
		yield tuple(rect)
		rect[incindex] += (incindex >= 2) and 1 or -1
		count += 1
		if count >= maxCount: return
		
def dockRectProbs(im, x1, y1, x2, y2, incindex, maxCount = None, minSize = None):
	if not maxCount: maxCount = max(im.width,im.height)
	dockrects = [DockRect(im, (_x1,_y1,_x2,_y2)) for _x1,_y1,_x2,_y2 in iterateRect(x1,y1,x2,y2, im.width, im.height, incindex, maxCount)]
	return [ dockrect.probability([incindex], minSize, surroundingSpace=30) for dockrect in dockrects ]

def normProbsWithParams(probs):
	if len(probs) == 0: return (probs, 0, 0)
	_max = max(probs)
	_min = min(probs)
	f = _max - _min
	if f == 0: return (probs, 0, 0)
	return ([ (x - _min) / f for x in probs ], _min, _max)

def normProbs(probs):
	return normProbsWithParams(probs)[0]
	
def probToColor(p):
	return cv.RGB(p * 255, p * 255, p * 255)

def argmax(list, index=0):
	if index == 0:
		return list.index(max(list))
	else:
		return list.index(max(list, key = itemgetter(index)))

def estimated_argmax(list, misscountMax = 1000):
	i = 0
	m = None
	mi = 0 # None
	misscount = 0	
	for o in list:
		if m == None or o > m:
			mi = i
			m = o
			misscount = 0
		elif misscount > misscountMax: break
		i += 1
	return mi


def bestRectCoordWithProb(im, x1,y1,x2,y2, index):
	probs = list(dockRectProbs(im, x1,y1,x2,y2, index))
	i = estimated_argmax(probs)
	return (i, probs[i])

def bestDockBottom(im):
	y1,y2 = im.height-1, im.height
	dx1,pdx1 = bestRectCoordWithProb(im, im.width/2,y1,im.width/2,y2, 0)
	x1 = im.width/2 - dx1
	dx2,pdx2 = bestRectCoordWithProb(im, im.width/2,y1,im.width/2,y2, 2)
	x2 = im.width/2 + dx2
	return ((x1,y1,x2,y2), pdx1 * pdx2)

def bestDockVert(im, x):
	x1,x2 = x, x+1
	dy1,pdy1 = bestRectCoordWithProb(im, x1,im.height/2,x2,im.height/2, 1)
	y1 = im.height/2 - dy1
	dy2,pdy2 = bestRectCoordWithProb(im, x1,im.height/2,x2,im.height/2, 3)
	y2 = im.height/2 + dy2
	return ((x1,y1,x2,y2), pdy1 * pdy2)

def bestDockLeft(im):
	return bestDockVert(im, 0)

def bestDockRight(im):
	return bestDockVert(im, im.width-1)



def makeSquare(rect, newsize = None):
	x1,y1,x2,y2 = rect
	w = x2 - x1
	h = y2 - y1
	if not newsize: newsize = max(w, h)
	#if not newsize: newsize = (w + h) / 2
	x1 -= (newsize - w) / 2
	x2 += (newsize - w) / 2
	y1 -= (newsize - h) / 2
	y2 += (newsize - h) / 2
	return (x1,y1,x2,y2)

def iterateRectSet(baserect, dist, index, rectCount = 4):
	r = list(baserect)
	D = r[index] - r[index-2]
	num = 0
	while True:
		if rectCount >= 0 and num >= rectCount: return
		yield tuple(r)
		r[index-2] += D + dist
		r[index] = r[index-2] + D
		num += 1

def probabilityOfRectset(im, baserect, dist, index, rectCount):
	minSize = 400
	probargs = (range(0,4), minSize, 1)
	return sum(
		map(lambda r: DockRect(im, r).probability(*probargs),
			iterateRectSet(baserect, dist, index, rectCount)) )

def filterGoodIconRects(im, rects):
	borderSpace = 12
	minSize = 5
	for r in rects:
		x1,y1,x2,y2 = r
		if x2 - x1 < minSize: continue
		if y2 - y1 < minSize: continue
		if x1 < borderSpace or x2 >= im.width - borderSpace: continue
		if y1 < borderSpace or y2 >= im.height - borderSpace: continue
		yield r

def rectCenter(r):
	x1,y1,x2,y2 = r
	return ((x1+x2)/2, (y1+y2)/2)

def bestSquareRects(im, x1,y1,x2,y2, index):
	rectCount = 4
	minSize = 200
	step = 5
	dist = 10
	while True:
		size = x2-x1
		oldrect = (x1,y1,x2,y2)
		x,y = rectCenter(oldrect)

		baserects = []
		for s in xrange(size+step*step,size-1,-1):
			for dx in xrange(-step,step):
				for dy in xrange(-step*2,step*2):
					baserects += [(x+dx-s/2,y+dy-s/2,x+dx+s/2,y+dy+s/2)]
		#baserects = map(makeSquare, baserects)
		baserects = filterGoodIconRects(im, baserects)
		baserects = set(baserects)
		dockrects = []
		for r in baserects:
			for d in range(max(0, dist - 4*step), dist + 4*step, 1):
				dockrects += [(r, d, probabilityOfRectset(im, r, d, index, rectCount))]
		
		bestrect,dist,bestprob = max(dockrects, key = itemgetter(2))
		x1,y1,x2,y2 = bestrect
		size = x2-x1
		#print oldrect, bestrect, rectSize(oldrect), rectSize(bestrect), dist, bestprob, len(dockrects)
		if bestrect[0] == oldrect: break
		#if rectSize(bestrect[0]) >= minSize and rectSize(bestrect[0]) - rectSize(oldrect) <= 0: break
		step -= 1
		#rectCount += 1
		if step < 1: break
	return (bestrect,dist,bestprob)




def iterateIconsMostProbable(im, baserect, dist, index):
	probs = []
	forwardNum = 3
	for r in iterateRectSet(baserect, dist, index, -1):
		prob = probabilityOfRectset(im, r, dist, index, forwardNum) / forwardNum
		if prob == 0:
			print r, rectSize(r), dist, index, forwardNum, prob
			return
		
		probs += [probabilityOfRectset(im, r, dist, index, 1)]
		
		num = len(probs)
		#print num, ":", vecAverage(probs), prob, vecAverage(probs) / prob
		if vecAverage(probs) / prob > 1.4: return
		#if num > 20: return
		
		yield r
		

#files = glob("2010-10-*.png")
files = glob("2010-10-11.*.png") # bottom dock with eclipse
#files = glob("2010-10-28.*.png") # left dock with eclipse
i = 0
random.shuffle(files)

def showImageWithRects(im, rects):
	imcopy = cv.CloneImage(im)
	draw_rects(imcopy, rects)
	cv.ShowImage("Screenshot", imcopy)
	cv.WaitKey(1)

def vecMult(r, f):
	return map(lambda x: x * f, r)

def vecSum(r1, r2):
	return map(lambda (x1,x2): x1 + x2, zip(r1,r2))

def vecAverage(v):
	return sum(v) / len(v)

def rectMultSize(rect, f):
	x1,y1,x2,y2 = rect
	w = x2 - x1
	h = y2 - y1
	neww = w * f
	newh = h * f
	x1 -= (neww - w) / 2
	x2 += (neww - w) / 2
	y1 -= (newh - h) / 2
	y2 += (newh - h) / 2
	return (x1,y1,x2,y2)	
	
showProbs = False
while True:
	f = files[i]
	print f
	im = cv.LoadImage(f)

	rects = []
	showImageWithRects(im, rects)		


	(x1,y1,x2,y2),prob = bestDockLeft(im)
	print "left prob:", prob
	(x1,y1,x2,y2),prob = bestDockBottom(im)
	print "bottom prob:", prob

	rects += [(x1,y1,x2,y2)]
	dockx1,dockx2 = x1,x2
	showImageWithRects(im, rects)
	
	index = 2
	rect,dist,prob = bestSquareRects(im, x1 + 30, im.height - 20, x1 + 35, im.height - 15, index)
	rects += iterateIconsMostProbable(im,rect,dist,index)
	
	if showProbs:
		x = im.width/2
		for p in dockRectProbs(im, im.width/2,im.height-1,im.width/2,im.height, 2):
			print x, ":", p
			draw_rects(im, [(x-1,im.height-1,x,im.height)], probToColor(p))
			x += 1
		rects = []
		
	showImageWithRects(im, rects)		
	RectProbCache = dict()
	im = None
	rects = None
	
	key = cv.WaitKey(0)
	if key in [27, ord('q')]: quit()
	elif key == 63235: i = min(i + 1, len(files))
	elif key == 63234: i = max(i - 1, 0)
	elif key == ord('p'): showProbs = not showProbs
	else: print "key", key, "unknown"
	