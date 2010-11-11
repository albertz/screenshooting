#!/usr/bin/python

import cv
from glob import glob
from math import *
import random

cv.NamedWindow('Screenshot', cv.CV_WINDOW_AUTOSIZE)

def draw_rects(im, rects, color = cv.RGB(0,255,0)):
	for x1,y1,x2,y2 in rects:
		cv.Rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)),
					 color, 3, 8, 0)

def random_colors_in_aray(im, rect):
	pass

h_bins = 40
s_bins = 40

def hist_image(hist):
	(_, max_value, _, _) = cv.GetMinMaxHistValue(hist)
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

def __hs_histogram_base(src, hist):
	srcsize = cv.GetSize(src)
	
	# Convert to HSV
	hsv = cv.CreateImage(srcsize, 8, 3)
	cv.CvtColor(src, hsv, cv.CV_RGB2HSV)
	
	# Extract the H and S planes
	h_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	s_plane = cv.CreateMat(srcsize[1], srcsize[0], cv.CV_8UC1)
	cv.Split(hsv, h_plane, s_plane, None, None)
	planes = [h_plane, s_plane]
	
	cv.CalcHist([cv.GetImage(i) for i in planes], hist, 1)

def __create_default_hist():
	# hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
	h_ranges = [0, 180]
	# saturation varies from 0 (black-gray-white) to
	# 255 (pure spectrum color)
	s_ranges = [0, 255]
	ranges = [h_ranges, s_ranges]
	hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
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
	cv.NormalizeHist(hist, 1.0)
	return hist

def hist_for_im_rect(im, rect):
	return hist_for_im_rects(im, [rect])

def hist_for_horizline(im, x1, y, x2):
	return hist_for_im_rect(im, (x1, y, x2, y + 1))

def hist_for_vertline(im, x, y1, y2):
	return hist_for_im_rect(im, (x, y1, x + 1, y2))

def hist_for_baseline(im, x1, x2):
	return hist_for_horizline(im, x1, im.height - 1, x2)

def hist_for_line(im, x1, x2, y1, y2):
	if x1 == x2: return hist_for_vertline(im, x1, y1, y2)
	if y1 == y2: return hist_for_horizline(im, x1, y1, x2)
	assert False

def __random_divide(num, partsnum = 3):
	fracs = [ random.random() for i in xrange(0, partsnum) ]
	s = sum(fracs)
	return [ f * num / s for f in fracs ]

def random_divide(start, end, partsnum = 3):
	return [ start + x for x in __random_divide(end - start, partsnum) ]

def make_sequences(seps):
	return [ (seps[i], seps[i+1]) for i in xrange(0, len(seps) - 1) ]

def random_sequences(start, end, sepnum = 3):
	return make_sequence( [start] + random_divide(start, end, sepnum) )
	
def random_line_seps(im, x1, x2, y1, y2, sepnum = 3):
	if x1 == x2: return [ (x1, _y1, x2, _y2) for _y1,_y2 in random_sequence(y1,y2) ]
	if y1 == y2: return [ (_x1, y1, _x2, y2) for _x1,_x2 in random_sequence(x1,x2) ]
	assert False


def index_filter(ls, *indices):
	return [ ls[i] for i in indices ]

def rectSize(rect):
	return (rect[2] - rect[0]) * (rect[3] - rect[1])
	
def rectsSizeSum(rects):
	return sum(rectSize(rect) for rect in rects)

class DockRect:
	def __init__(self, im, rect):
		self.im = im
		self.rect = rect
	
	def surrounding_rects(self):
		space = 100
		x1,y1,x2,y2 = self.rect
		r1 = (x1 - space, y1, x1, y2) # left
		r2 = (x1 - space, y1 - space, x2 + space, y1) # top
		r3 = (x2, y1, x2 + space , y2) # right
		r4 = (x1 - space, y2, x2 + space, y2 + space) # bottom
		return [ r1, r2, r3, r4 ]
	
	def inner_rects(self):
		thick = 1
		x1,y1,x2,y2 = self.rect
		r1 = (x1,y2-thick,x2,y2)
		r2 = (x1,y1,x1+thick,y2)
		r3 = (x2-thick,y1,x2,y2)
		#if x1 == 0: return [r1, r2]
		#return [r1, r3]
		return [r1]
		
	def probability(self, indices = xrange(0,4)):
		innerRects = self.inner_rects()
		outerRects = index_filter(self.surrounding_rects(), *indices)
		if rectsSizeSum(innerRects) < 30: return 0
		if rectsSizeSum(outerRects) < 30: return 0
		histInner = hist_for_im_rects(self.im, innerRects)
		histOuter = hist_for_im_rects(self.im, outerRects)
		histDiff = cv.CompareHist(histInner, histOuter, cv.CV_COMP_BHATTACHARYYA)
		#histDiff = -histDiff
		#print histDiff
		return histDiff # the higher the diff, the better the probalitity (we want to have the best sepeaation)

def best_avg_dockrect(dockrects, indices):
	dockrects = [ (dockrect.rect, dockrect.probability(indices)) for dockrect in dockrects ]
	probsum = sum([ p for _,p in dockrects ])
	rect = [0,0,0,0]
	for dockrect,dockprob in dockrects:
		for i in [0,1,2,3]:
			rect[i] += dockrect[i] * dockprob / probsum
	return tuple(rect)

def best_dockrect(dockrects, indices):
	dockrects = [ (dockrect.rect, dockrect.probability(indices)) for dockrect in dockrects ]
	probmax = -100000000
	rect = (0,0,0,0)
	for dockrect,dockprob in dockrects:
		if dockprob >= probmax:
			probmax = dockprob
			rect = dockrect
	print rect, probmax
	return rect

def best_dockrect__cutoff(dockrects, indices, cutoffnum = 100):
	dockrects = [ (dockrect.rect, dockrect.probability(indices)) for dockrect in dockrects ]
	probmax = -100000000
	rect = (0,0,0,0)
	misscount = 0
	for dockrect,dockprob in dockrects:
		if dockprob > probmax:
			misscount = 0
			probmax = dockprob
			rect = dockrect
		elif probmax > 0.5:
			misscount += 1
			if misscount > cutoffnum: break
	return rect

def random_dockrect_bottom(im, inrect):
	y1 = random.uniform(inrect[1], inrect[3])
	y2 = inrect[3]
	x1, x2, _ = random_divide(inrect[0], inrect[2], 3)
	return DockRect(im, (x1,y1,x2,y2))


def iterateRect(x1,y1,x2,y2, maxx, maxy, incindex, maxCount):
	rect = [x1,y1,x2,y2]
	earlierBreak = 100
	count = 0
	while True:
		x1,y1,x2,y2 = rect
		if x1 < 0 or x1 >= maxx: return
		if x2 < 0 or x2 > maxx: return
		if y1 < 0 or y1 >= maxy: return
		if y2 < 0 or y2 > maxy: return
		if incindex == 0 and rect[incindex] < earlierBreak: return
		if incindex == 2 and rect[incindex] >= maxx - earlierBreak: return
		if incindex == 1 and rect[incindex] < earlierBreak: return
		if incindex == 3 and rect[incindex] >= maxy - earlierBreak: return 
		yield tuple(rect)
		rect[incindex] += (incindex >= 2) and 1 or -1
		count += 1
		if count >= maxCount: return
		
def dockRectProbs(im, x1, y1, x2, y2, incindex, maxCount = None):
	if not maxCount: maxCount = max(im.width,im.height)
	dockrects = [DockRect(im, (_x1,_y1,_x2,_y2)) for _x1,_y1,_x2,_y2 in iterateRect(x1,y1,x2,y2, im.width, im.height, incindex, maxCount)]
	return [ dockrect.probability([incindex]) for dockrect in dockrects ]

def normProbs(probs):
	if len(probs) == 0: return probs
	_max = max(probs)
	_min = min(probs)
	f = _max - _min
	if f == 0: return probs
	return [ (x - _min) / f for x in probs ]

def probToColor(p):
	return cv.RGB(p * 255, p * 255, p * 255)

def argmax(list):
	return list.index(max(list))
	i = 0
	m = None
	mi = None
	for o in list:
		if m == None or o > m:
			mi = i
			m = o
		i += 1
	return mi

def estimated_argmax(list):
	i = 0
	m = None
	mi = None
	misscount = 0	
	for o in list:
		if m == None or o > m:
			mi = i
			m = o
			misscount = 0
		elif misscount > 100: break
		i += 1
	return mi


def bestDockX1(im):
	x1,y1,x2,y2 = im.width/2, im.height-1, im.width/2, im.height
	x1 -= estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 0))
	return x1

def bestDockX2(im):
	x1,y1,x2,y2 = im.width/2, im.height-1, im.width/2, im.height
	x2 += estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 2))
	return x2

def bestDockY1(im, x1, x2):
	y1,y2 = im.height-30, im.height
	y1 -= estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 1))
	return y1



files = glob("2010-10-11.*.png") # bottom dock with eclipse
#files = glob("2010-10-28.*.png") # left dock with eclipse
i = 0

showProbs = False
while True:
	f = files[i]
	print f
	im = cv.LoadImage(f)

	cv.ShowImage('Screenshot', im)
		
	x1,y1,x2,y2 = im.width/2, im.height-1, im.width/2, im.height
	x1 = bestDockX1(im)
	x2 = bestDockX2(im)
	y1 = bestDockY1(im, x2 - 20, x2)

	if showProbs:
		x = x1
		#for p in normProbs(dockRectProbs(im, im.width/2, im.height-1, im.width/2, im.height, 2)):
		for p in normProbs(dockRectProbs(im, x1,y1,x2,y2, 0)):
			draw_rects(im, [(x,im.height-2,x+1,im.height)], probToColor(p))
			x -= 1
		rect = None
	else:
		rect = (x1,y1,x2,y2)
		
		#rect = best_dockrect([DockRect(im, (im.width/2,im.height-1,x,im.height)) for x in xrange(im.width/2, im.width-100)], [2])
	
	#rect = best_dockrect([random_dockrect_bottom(im, (0,im.height-1,im.width,im.height)) for i in xrange(0,100)])
	#print rect
	if rect != None: draw_rects(im, [rect])
	cv.ShowImage('Screenshot', im)
	
	key = cv.WaitKey(0)
	if key in [27, ord('q')]: quit()
	elif key == 63235: i = min(i + 1, len(files))
	elif key == 63234: i = max(i - 1, 0)
	elif key == ord('p'): showProbs = not showProbs
	else: print "key", key, "unknown"
	