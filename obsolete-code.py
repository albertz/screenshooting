#!/usr/bin/python


def iterateResizedRectFull(x1,y1,x2,y2, count):
	yield (x1,y1,x2,y2)
	while count > 0:
		x2 += 1
		y2 += 1
		yield (x1,y1,x2,y2)
		x1 -= 1
		y1 -= 1
		yield (x1,y1,x2,y2)
		count -= 1
		
def iterateMovedRectFull(x1,y1,x2,y2, count):
	orig = (x1,y1,x2,y2)
	yield orig
	x1 -= count
	x2 -= count
	y1 -= count
	y2 -= count
	while count > 0:
		for i in xrange(count * 2):
			x1 += 1
			x2 += 1
			yield (x1,y1,x2,y2)
		for i in xrange(count * 2):
			y1 += 1
			y2 += 1
			yield (x1,y1,x2,y2)
		for i in xrange(count * 2):
			x1 -= 1
			x2 -= 1
			yield (x1,y1,x2,y2)
		for i in xrange(count * 2):
			y1 -= 1
			y2 -= 1
			yield (x1,y1,x2,y2)
		x1 += 1
		x2 += 1
		y1 += 1
		y2 += 1
		count -= 1


def bestDockY1(im, x1, x2):
	y1,y2 = im.height-30, im.height
	y1 -= estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 1))
	return y1

def bestRect(im, x1,y1,x2,y2, maxCount = None):
	if not maxCount: maxCount = max(im.width, im.height)
	x1 -= estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 0))
	y1 -= estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 1))
	x2 += estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 2))
	y2 += estimated_argmax(dockRectProbs(im, x1,y1,x2,y2, 3))
	return (x1,y1,x2,y2)


def bestSquareRect(im, x1,y1,x2,y2, maxCount = None):
	if not maxCount: maxCount = max(im.width, im.height)
	minSize = 200
	while True:
		oldrect = (x1,y1,x2,y2)

		dockrects = [(x1,y1,x2,y2)]
		for i in range(0,4):
			dockrects += iterateRect(x1,y1,x2,y2, im.width, im.height, i, 30, 10)
		dockrects = map(makeSquare, dockrects)

		if False:
			dockrects = iterateResizedRectFull(x1,y1,x2,y2, count=2)
			dockrects = map(lambda rect: iterateMovedRectFull(*rect, count=5), dockrects)
			dockrects = set(chain(*dockrects))
		
		dockrects = [ DockRect(im, rect) for rect in dockrects ]

		bestrect = best_dockrect(dockrects, range(0,4), minSize, 10)
		x1,y1,x2,y2 = bestrect
		print oldrect, bestrect, rectSize(oldrect), rectSize(bestrect), len(dockrects)
		if bestrect == oldrect: break
		if rectSize(bestrect) >= minSize and rectSize(bestrect) - rectSize(oldrect) <= 0: break
	return (x1,y1,x2,y2)




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

def __hs_histogram_base__hs(src, hist):
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

def __create_default_hist__hs():
	# hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
	h_ranges = [0, 180]
	# saturation varies from 0 (black-gray-white) to
	# 255 (pure spectrum color)
	s_ranges = [0, 255]
	ranges = [h_ranges, s_ranges]
	hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
	cv.ClearHist(hist)
	return hist




def partition(list, left, right, pivotIndex):
	pivotValue = list[pivotIndex]
	list[pivotIndex], list[right] = list[right], list[pivotIndex] # Move pivot to end
	storeIndex = left
	for i in range(left, right):
		if list[i] < pivotValue:
			list[storeIndex], list[i] = list[i], list[storeIndex]
			storeIndex += 1
	list[right], list[storeIndex] = list[storeIndex], list[right] # Move pivot to its final place
	return storeIndex
	 
def quickfindFirstK(list, left, right, k):
	if right > left:
		# select pivotIndex between left and right
		pivotIndex = (right - left) / 2		
		pivotNewIndex = partition(list, left, right, pivotIndex)
		if pivotNewIndex > k: # new condition
			quickfindFirstK(list, left, pivotNewIndex-1, k)
		if pivotNewIndex < k: # questionable
			quickfindFirstK(list, pivotNewIndex+1, right, k)
