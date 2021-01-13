import numpy as np
import cv2

cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
ret,frame = cap.read()
cpt=0

while ret == True:
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 3x3 sobel filters for edge detection
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Filter the blurred grayscale images using filter2D # TODO sobel functions ?
    filtered_x = cv2.filter2D(frame1, cv2.CV_32F, sobel_x)
    filtered_y = cv2.filter2D(frame1, cv2.CV_32F, sobel_y)

    # Compute the orientation of the image
    orien = cv2.phase(np.array(filtered_x, np.float32),
                        np.array(filtered_y, np.float32),
                        angleInDegrees=True)
    mag = cv2.magnitude(filtered_x, filtered_y)


    # thresholding of the magnitude values, play with the thresh value adjust it too your liking
    thresh = 50
    _, mask = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)

    image_map = np.zeros((orien.shape[0], orien.shape[1], 3),
							 dtype=np.float32)


    image_map = np.zeros((orien.shape[0], orien.shape[1], 3),
							 dtype=np.float32)

    # Define RGB colours
    red = np.array([0, 0, 255])
    image_map[:,:,0]=orien
    image_map[:,:,1]=orien
    image_map[:,:,2]=orien

    # Set colours corresponding to angles
    image_map[(mask == 0)] = red

    cv2.imshow("frame",frame)
    cv2.imshow("orientation",orien)
    cv2.imshow("gradient",mag)
    cv2.imshow("image_map",image_map)
    k=cv2.waitKey(25)
    if k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%cpt,frame)
        cv2.imwrite('Ori_%04d.png'%cpt,orien)
        cv2.imwrite('Gra_%04d.png'%cpt,mag)
        cv2.imwrite('map_%04d.png'%cpt,image_map)
    cpt += 1

    ret,frame = cap.read()


cv2.destroyAllWindows()
cap.release()
