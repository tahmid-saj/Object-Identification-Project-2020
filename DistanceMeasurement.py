# Import the relevant libraries, packages, methods etc
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


def finddistance(imgOrig, imgOrigGray):
    # Define the midpoint of two points
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # Find the blur of the imgOrigGray image
    imgOrigGray = cv2.GaussianBlur(imgOrigGray, (7, 7), 0)

    # Determine the width of the reference image
    width = 20

    # Perform edge detection, then perform a dilation plus erosion to
    # close gaps in between object edges
    edged = cv2.Canny(imgOrigGray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
    refObj = None

    # Loop over the contours individually
    for c in cnts:
        # If the contour area is not large, then ignore it
        if cv2.contourArea(c) < 200:
            continue

        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)

        # Compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # If this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the reference object
        if refObj is None:
            # Unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / width)
            continue

        # Draw the contours on the image
        orig = imgOrig.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

        # Stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        # Loop over the original points
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            # Draw circles corresponding to the current points and
            # connect them with a line
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)

            # Compute the Euclidean distance between the coordinates,
            # and then convert the distance in pixels to distance in units
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            #  Display the distances on the image
            cv2.imshow("Detected Object - Distances", orig)

            # If any key is pressed but (esc, s, d, n), then the next image will show
            # If esc key is pressed, then the program will shut down
            key = cv2.waitKey(0) & 0xFF

            if key == ord('d'):
                continue
