#!/usr/bin/python

import cv
import numpy as np
import scipy.spatial as sp
from glob import glob
import math

def bwImage(im):
	if len(im.channels) == 1: return im
	return cv.cvtColor(im, cv.CV_RGB2GRAY)

class MatchableImage:
	def __init__(self, im):
		self.image = im
		(self.keypoints, self.descriptors) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, 300, 3, 4))
		#for ((x, y), laplacian, size, dir, hessian) in self.keypoints:
			#print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)
		#print self.descriptors
		
	def findFeatures(self, alg_name):
		detector = cv.createFeatureDetector(alg_name)
		self.features = detector.detect(bwImage(self.image))
		
	def findDescriptors(self, alg_name):
		grayim = bwImage(self.image)
		descriptorExtractor = cv.createDescriptorExtractor(alg_name)
		self.descriptors = descriptorExtractor.compute(grayim, features)
		
	def trainMatcher(self, alg_name): pass
	def matchTo(self, im):
		pass


def compareSURFDescriptors(d1, d2, best, length):
	total_cost = 0
	assert( length % 4 == 0 )
	for i in xrange(0, length, 4):
		t0 = d1[i] - d2[i]
		t1 = d1[i+1] - d2[i+1]
		t2 = d1[i+2] - d2[i+2]
		t3 = d1[i+3] - d2[i+3]
		total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3
		if total_cost > best:
			break
	return total_cost

def naiveNearestNeighbor(vec, laplacian, model_keypoints, model_descriptors):
	length = model_descriptors.elem_size/sizeof(c_float)
	neighbor = -1
	dist1 = 1e6
	dist2 = 1e6
	kp_arr = model_keypoints.asarray(CvSURFPoint)
	mv_arr = model_descriptors.asarrayptr(POINTER(c_float))

	for i in xrange(model_descriptors.total):
		if  laplacian != kp_arr[i].laplacian:
			continue
		d = compareSURFDescriptors(vec, mv_arr[i], dist2, length)
		if d < dist1:
			dist2 = dist1
			dist1 = d
			neighbor = i
		elif d < dist2:
			dist2 = d

	if dist1 < 0.6*dist2:
		return neighbor
	return -1

def findPairs(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors):
	ptpairs = []
	kp_arr = objectKeypoints.asarray(CvSURFPoint)
	de_arr = objectDescriptors.asarrayptr(POINTER(c_float))

	for i in xrange(objectDescriptors.total):
		nn = naiveNearestNeighbor( de_arr[i], kp_arr[i].laplacian, imageKeypoints, imageDescriptors );
		if nn >= 0:
			ptpairs.append((i,nn))

	return ptpairs

# a rough implementation for object location
def locatePlanarObject(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, src_corners, dst_corners):
	ptpairs = findPairs(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors)
	n = len(ptpairs)
	if n < 4:
		return 0

	ok_arr = objectKeypoints.asarray(CvSURFPoint)
	pt1 = cvCreateMatFromCvPoint2D32fList([ok_arr[x[0]].pt for x in ptpairs])
	ik_arr = imageKeypoints.asarray(CvSURFPoint)
	pt2 = cvCreateMatFromCvPoint2D32fList([ik_arr[x[1]].pt for x in ptpairs])
	try:
		h = cvFindHomography( pt1, pt2, method=CV_RANSAC, ransacReprojThreshold=5 )[0]
	except RuntimeError:
		return 0

	for i in xrange(4):
		x = src_corners[i].x
		y = src_corners[i].y
		Z = 1./(h[2,0]*x + h[2,1]*y + h[2,2])
		X = (h[0,0]*x + h[0,1]*y + h[0,2])*Z
		Y = (h[1,0]*x + h[1,1]*y + h[1,2])*Z
		dst_corners[i] = cvPoint(cvRound(X), cvRound(Y))

	return 1

iconim = MatchableImage(cv.LoadImageM("eclipse-icon.png", cv.CV_LOAD_IMAGE_GRAYSCALE))
#iconim.findFeatures("SURF")
#iconim.trainMatcher("SURF")

def clone(something):
	if something.__class__ == cv.cvmat:
		return cv.CloneMat(something)
	else:
		return cv.CloneImage(something)

def draw_surf2(image, keypoints, colors):
	rimage = clone(image)
	for i, k in enumerate(keypoints):
		loc, lap, size, d, hess = k
		loc = tuple(np.array(np.round(loc), dtype='int').tolist())
		c = tuple(np.matrix(colors[:,i],dtype='int').T.A1)
		color = (int(c[0]), int(c[1]), int(c[2]))
		#cv.Circle(rimage, loc, int(round(size/2.)), color, 1, cv.CV_AA)
		cv.Circle(rimage, loc, 5, color, 1, cv.CV_AA)
	return rimage

def draw_surf(image, keypoints, color):
	rimage = clone(image)

	for loc, lap, size, d, hess in keypoints:
		loc = tuple(np.array(np.round(loc), dtype='int').tolist())
		circ_rad = int(round(size/4.))
		cv.Circle(rimage, loc, circ_rad, color, 1, cv.CV_AA)
		cv.Circle(rimage, loc, 2, color, -1, cv.CV_AA)

		drad = math.radians(d)
		line_len = circ_rad
		loc_end = (np.matrix(np.round( circ_rad * np.matrix([np.cos(drad), np.sin(drad)]).T + np.matrix(loc).T), dtype='int')).A1.tolist()
		cv.Line(rimage, loc, tuple(loc_end), color, thickness=1, lineType=cv.CV_AA)

	return rimage

def grayscale(image):
	image_gray = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U,1)
	cv.CvtColor(image, image_gray, cv.CV_BGR2GRAY)
	return image_gray

def surf(image_gray, params=(1, 300,3,4)):
	surf_stor = cv.CreateMemStorage()
	keypoints, desc = cv.ExtractSURF(image_gray, None, surf_stor, params)
	del surf_stor
	return ([loc for loc, lap, size, d, hess in keypoints], desc)

class SURFMatcher:
	def __init__(self):
		self.model_images = {}
		self.model_fea = {}

	def add_file(self, model_name, label):
		model_img = cv.LoadImage(model_name)
		self.add_model(model_img, label)

	def add_model(self, model_img, label):
		mgray = grayscale(model_img)
		m_loc, m_desc = surf(mgray)
		self.model_images[label] = model_img
		self.model_fea[label] = {'loc': m_loc, 'desc': m_desc}

	def build_db(self):
		fea_l = []
		labels_l = []
		locs_l = []
		for k in self.model_fea:
			desc = self.model_fea[k]['desc']
			fea_l.append(np.array(desc))
			loc = self.model_fea[k]['loc']
			locs_l.append(np.array(loc))
			labels_l.append(np.array([k for i in range(len(desc))]))

		self.labels = np.row_stack(labels_l)
		self.locs = np.row_stack(locs_l)
		self.tree = sp.KDTree(np.row_stack(fea_l))

	def match(self, desc, threshold=.6):
		dists, idxs = self.tree.query(np.array(desc), 2)
		ratio = dists[0] / dists[1]
		ratio = min(ratio)
		if ratio > threshold:
			desc = self.tree.data[idxs[0]]
			loc = self.locs[idxs[0]]
			return desc, loc
		else:
			return None

matcher = SURFMatcher()
matcher.add_file("eclipse-icon.png", "eclipse")
matcher.build_db()


cascade = cv.Load("eclipse-icon.xml")


def detect(im):
	storage = cv.CreateMemStorage(0)
	rects = cv.HaarDetectObjects(im, cascade, storage, 1.1, 2, 0, (10, 10))
	if len(rects) > 0: return rects
	return None

	#return matcher.match(surf(im, (1, 3000, 3, 4))[1])

	#input = MatchableImage(im)
	#input.findFeatures("SURF")
	#input.findDescriptors("SURF")
	
	#matches = iconim.matchTo(input)
	#Mat img_corr;
	#drawMatches(input.bw(), input.features, logos[i].logo.bw(), logos[i].logo.features, matches, img_corr);
	#imshow(logos[i].name, img_corr);

def foo():
	src_corners = (()*4)((0,0), (object.width,0), (object.width, object.height), (0, object.height))
	dst_corners = (()*4)()
	correspond = cvCreateImage( cvSize(image.width, object.height+image.height), 8, 1 )
	cvSetImageROI( correspond, cvRect( 0, 0, object.width, object.height ) )
	cvCopy( object, correspond )
	cvSetImageROI( correspond, cvRect( 0, object.height, correspond.width, correspond.height ) )
	cvCopy( image, correspond )
	cvResetImageROI( correspond )
	if locatePlanarObject( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, src_corners, dst_corners ):
		for i in xrange(4):
			r1 = dst_corners[i%4]
			r2 = dst_corners[(i+1)%4]
			cvLine( correspond, cvPoint(r1.x, r1.y+object.height ), cvPoint(r2.x, r2.y+object.height ), colors[8] )
	
	ptpairs = findPairs(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors)
	for i in xrange(len(ptpairs)):
		r1 = CV_GET_SEQ_ELEM(CvSURFPoint, objectKeypoints, ptpairs[i][0])[0]
		r2 = CV_GET_SEQ_ELEM(CvSURFPoint, imageKeypoints, ptpairs[i][1])[0]
		cvLine( correspond, cvPointFrom32f(r1.pt), cvPoint(cvRound(r2.pt.x), cvRound(r2.pt.y+object.height)), colors[8] )
	
	cvShowImage( "Object Correspond", correspond )
	for i in xrange(objectKeypoints.total):
		r = CV_GET_SEQ_ELEM(CvSURFPoint, objectKeypoints, i )[0]
		cvCircle( object_color, cvPoint(cvRound(r.pt.x), cvRound(r.pt.y)), cvRound(r.size*1.2/9.*2), colors[0], 1, 8, 0 )
	
	cvShowImage( "Object", object_color )

haveWindow = False

#cv.ShowImage('Pic', draw_surf(iconim.image, iconim.keypoints, cv.RGB(0,255,0)))
#cv.WaitKey(0)
#cv.ShowImage('Pic', draw_surf2(iconim.image, iconim.keypoints, (cv.RGB(0,255,0))))
#cv.WaitKey(0)


for f in glob("2010-10-11.*.png"):
	im = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
	#cv.ShowImage('Pic', im)

	rects = detect(im)
	if rects != None:
		print f, ": yes"
	else:
		print f, ": no"
		
	if rects:
		for (x,y,w,h),n in rects:
			cv.Rectangle(im, (x,y), (x+w,y+h),
						 cv.RGB(0, 255, 0), 3, 8, 0)
			
		if not haveWindow:
			cv.NamedWindow('Pic', cv.CV_WINDOW_AUTOSIZE)
			haveWindow = True
		cv.ShowImage('Pic', im)
		cv.WaitKey(0)
	