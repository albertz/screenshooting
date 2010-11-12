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

