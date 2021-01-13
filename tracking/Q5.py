import numpy as np
import cv2
from collections import defaultdict 

roi_defined = False

def auto_canny(image, sigma=0.5):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True


def gradient_orien(gray_image):
    dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    orien = cv2.phase(np.array(dx, np.float32),
                        np.array(dy, np.float32),
                        angleInDegrees=True)
    # mag = cv2.magnitude(dx, dy)
    return orien


def create_r_table(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray)
    h, w = gray.shape
    centre = [int(h / 2),int(w / 2)]

    orien = gradient_orien(edges)
    
    # 180 labels in the R-table
    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            r_table[int(orien[i,j])].append([centre[0] - i, centre[1] - j])

    return r_table


def vote(r_table, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply canny operator to extract edges
    edges = auto_canny(gray)
    cv2.imshow("edges",edges)
    orien = gradient_orien(edges)

    accumulator = np.zeros(gray.shape)
    # accumulate the votes
    for (i, j), value in np.ndenumerate(edges):
        if value:
            r_array=np.array(r_table[int(orien[i,j])])
            if r_array.shape[0]<1:
                continue
            accum_i=r_array[:,0]+i
            accum_j=r_array[:,1]+j
            t=np.where((accum_i<accumulator.shape[0])&(accum_j < accumulator.shape[1])&(accum_i >= 0) & (accum_j >= 0))
            t=np.squeeze(t)
            accumulator[accum_i[t],accum_j[t]]+=1

    return accumulator


cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('Test-Videos/VOT-Woman.mp4')

# take first frame of the video
ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r, c, h, w)
# set up the ROI for tracking
roi = frame[c:c + w, r:r + h]
# get r-table
r_table = create_r_table(roi)
# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1
while (1):
    ret, frame = cap.read()
    if ret == True:
        accumulator = vote(r_table, frame)
        accumulator = accumulator/accumulator.max()
        ret, track_window = cv2.meanShift(accumulator, track_window, term_crit)
        
        # adaptive tracking window 
        # ret, track_window = cv2.CamShift(accumulator, track_window, term_crit)

        r,c,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        
        cv2.imshow('Sequence', frame_tracked)
        
        cv2.imshow('Meanshift', accumulator)

        # Save images
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Hough_%04d.png' % cpt, 255*accumulator)
            cv2.imwrite('Frame_%04d.png' % cpt, frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()