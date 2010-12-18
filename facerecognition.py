#!/usr/bin/python

# from: http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/

import sys
import cv
 
def detect(image):
    image_size = cv.GetSize(image)
 
    # create grayscale version
    grayscale = cv.CreateImage(image_size, 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
 
    # create storage
    storage = cv.CreateMemStorage(0)
    #cv.ClearMemStorage(storage)
 
    # equalize histogram
    cv.EqualizeHist(grayscale, grayscale)
 
    # detect objects
    #cascade = cv.LoadHaarClassifierCascade('haarcascade_frontalface_alt.xml', cv.Size(1,1))
    cascade = cv.Load("haarcascade_frontalface_alt.xml")
    faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (50, 50))
 
    if faces:
		face, _ = max(faces, key = lambda ((x,y,w,h),n) : w * h)
		x,y,w,h = face
		c = 30
		face = x - c, y - c, w + 2*c, h + 2*c 
		return face, faces
		
    return None, []
	

def resizedImage(im, w, h):
	newim = cv.CreateImage((w,h), cv.IPL_DEPTH_8U, im.channels)
	interpol = cv.CV_INTER_CUBIC
	#interpol = CV_INTER_LINEAR
	#interpol = CV_INTER_AREA
	cv.Resize(im, newim, interpol)
	return newim

def subImageScaled(im, rect, w, h):
	rect = (rect[0], rect[1], rect[2], rect[3])
	cv.SetImageROI(im, rect)
	resizedim = resizedImage(im, w, h)
	return resizedim
	
 
if __name__ == "__main__":
    #print "OpenCV version: %s (%d, %d, %d)" % (cv.CV_VERSION,
    #                                           cv.CV_MAJOR_VERSION,
    #                                           cv.CV_MINOR_VERSION,
    #                                           cv.CV_SUBMINOR_VERSION)
 
    print "Press ESC to exit ..."
 
    # create windows
    #cv.NamedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)
 
    # create capture device
    device = 0 # assume we want first device
    capture = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)    
 
    # check if capture device is OK
    if not capture:
        print "Error opening capture device"
        sys.exit(1)
 
    while 1:
        # do forever
 
        # capture the current frame
        frame = cv.QueryFrame(capture)
        if frame is None:
            break
 
        # mirror
        cv.Flip(frame, None, 1)
 
        # face detection
        face, faces = detect(frame)
        if face:
            f = face[3]/face[2]
            # display webcam image
            cv.ShowImage('Face', subImageScaled(frame, face, 200, 200*f))
			
        cv.SetImageROI(frame, (0,0,frame.width,frame.height))
        for (x,y,w,h),n in faces:
            cv.Rectangle(frame, (x,y), (x+w,y+h), cv.RGB(0, 255, 0), 3, 8, 0)
        cv.ShowImage('Camera', frame)
 
        # handle events
        k = cv.WaitKey(10)
 
        if k == 0x1b: # ESC
            print 'ESC pressed. Exiting ...'
            break
