import cv2 
import numpy as np

def countObjects(contours, image, draw=False):
    largest_box = None
    largest_area = 50  # Minimum area threshold

    # Iterate through contours to find the one with the largest area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            largest_box = (x, y, w, h)

    # If a bounding box is found and draw is True, create and show a copy of the image with the bounding rectangle
    if largest_box and draw:
        rectangleImage = image.copy()
        x, y, w, h = largest_box
        cv2.rectangle(rectangleImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return largest_box, rectangleImage

    return largest_box, image  # Return the largest box or None, along with the original image


def findStopSign(img):
    hsv_lower = np.array([138, 125, 0])
    hsv_upper = np.array([179,255,255])

    
    image = cv2.imread(img)[115:,:]
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contorImage = image.copy()
    cv2.drawContours(contorImage, contours, -1, (0, 255, 0), 3)

    stop, stop_image = countObjects(contours,image, True)

    if stop != None:
        return True
    else:
        return False