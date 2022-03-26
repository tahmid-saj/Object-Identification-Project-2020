# Import the relevant libraries, packages, methods etc
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils


def findsize(imgOrig, imgOrigGray):
    # Define the midpoint of two points
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # Initialize the width of the reference image, in this case, 12.5 cm
    width = 12.5

    # Find the blur of the imgOrigGray image
    imgOrigBlur = cv2.GaussianBlur(imgOrigGray, (7, 7), 0)

    # Perform edge detection, then perform a dilation plus erosion to
    # close the gaps in between object edges
    edged = cv2.Canny(imgOrigBlur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find the contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left-to-right and initialize the
    # pixels per metric calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # Loop over the contours individually
    for c in cnts:
        # If the contour area is not larger, then ignore it
        if cv2.contourArea(c) < 500:
            continue

        # Compute the rotated bounding box of the contour
        orig = imgOrig.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # Order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (255, 0, 0), 2)

        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (255, 0, 0), -1)

        # Unpack the ordered bounding box, then compute the midpoints
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # Compute the midpoint between the top-left and top-right points
        # followed by the midpoint between the top-right and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        # Compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Draw the object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimA), (int(tltrX + 20), int(tltrY + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1)
        cv2.putText(orig, "{:.1f}cm".format(dimB), (int(trbrX - 30), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1)

        #  Display the object sizes on the image
        cv2.imshow("Detected Object - Sizes", orig)

        # If any key is pressed but (esc, s, d, n), then the next image will show
        # If esc key is pressed, then the program will shut down
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            continue
