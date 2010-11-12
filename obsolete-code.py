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
