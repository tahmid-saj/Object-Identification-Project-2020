# Import the relevant libraries, packages, methods etc
import numpy as np
import cv2


def processimage(imgOrig, imgOrigGray):
    # Define the segmentation
    sobelx = cv2.Sobel(imgOrig, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(imgOrig, cv2.CV_64F, 0, 1)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))

    segmentation = cv2.bitwise_or(sobelx, sobely)

    # Define the morphology
    _, mask = cv2.threshold(imgOrig, 165, 255, cv2.THRESH_BINARY)
    kernal = np.ones((5, 2), np.uint8)
    morph_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # Display the segmentation (edge locations), morphology and
    # return the greyscale background subtraction of the image
    horiz = np.hstack((segmentation, morph_opening))
    cv2.imshow("Image Processing - Segmentation/Morphology and Background Subtraction", horiz)

    #The rest of the code in this area is for returning
    # the greyscale background subtraction