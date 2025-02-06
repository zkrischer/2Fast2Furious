import cv2
import numpy as np
import glob

def nothing(x):
    pass

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


# Load image

def img_test(image):
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    print("hi")
    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0

def imges(images):
    result = []
    for image in images: 
        ses = cv2.imread(image)
        ses = cv2.resize(ses, (150,150))
        ses = ses[100:,:]
        result.append(ses)
        

    
    img = cv2.hconcat(result)
    
    
    img_test(img)


def findStopSign(images):
    hsv_lower = np.array([138, 125, 0])
    hsv_upper = np.array([179,255,255])

    for img in images:
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

        ims = cv2.hconcat([image,result,contorImage,stop_image])
        cv2.imshow("test",ims)
        cv2.waitKey(0)



def main():
    images = glob.glob('train_kat/*.jpg')   
    findStopSign(images)
    



if __name__ == "__main__":
    main()