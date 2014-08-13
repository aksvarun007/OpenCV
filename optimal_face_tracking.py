import cv2
import cv2.cv as cv
import numpy as np

   
def camshift_tracking(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        return track_window
 

cap = cv2.VideoCapture(0)
ret,img = cap.read()
face_cascade=cv2.CascadeClassifier(sys.arg[1])//enter the location of the haar cascade xml file of your choice at this location.Found in data directory in opencv
faces=face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(30,30), flags = cv.CV_HAAR_SCALE_IMAGE)

#(x,y,w,h)=(faces.item(0),faces.item(1),faces.item(2),faces.item(3))
#bb=(x,y,w,h)
bb = (125,125,200,100) # get bounding box from some method
while True:
   
	ret,img1 = cap.read()
        bb = camshift_tracking(img1, img, bb)
        (x,y,w,h)=bb
        #print x,y
        img = img1
                #draw bounding box on img1
        #print bb
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("CAMShift",img1)
        k=cv2.waitKey(10)
        if k==27:
        	break
cv2.destroyAllWindows()
