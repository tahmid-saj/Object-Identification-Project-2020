# Import the relevant libraries, packages, methods etc
import cv2
import numpy as np
import glob
import imutils
import ObjectMeasurement
import DistanceMeasurement
import ImageProcessing

# Object detection part-------------------------------------------------------------------------------------------------

# Load Yolo files
net_1 = cv2.dnn.readNet("yolov3_wipe_training_last.weights", "yolov3_testing.cfg")
net_2 = cv2.dnn.readNet("yolov3_leaf_training_last.weights", "yolov3_testing.cfg")
net_3 = cv2.dnn.readNet("yolov3_twig_training_last.weights", "yolov3_testing.cfg")

# Name the custom object class
classes = ["Wipe", "Leaf", "Twig"]

# Declare the images path
images_path = glob.glob(r"F:\Imaging Research Project Files\Wipe\*.jpg")

# Declare the layer names for each object class
layer_names_1 = net_1.getLayerNames()
output_layers_1 = [layer_names_1[i[0] - 1] for i in net_1.getUnconnectedOutLayers()]

layer_names_2 = net_2.getLayerNames()
output_layers_2 = [layer_names_2[i[0] - 1] for i in net_2.getUnconnectedOutLayers()]

layer_names_3 = net_3.getLayerNames()
output_layers_3 = [layer_names_3[i[0] - 1] for i in net_3.getUnconnectedOutLayers()]

# Randomize the colours of the object detected box
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loop through all the images in the image path folder selected
for img_path in images_path:
    # Load the image and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.2, fy=0.14)

    # Find the height, width, and channels of the image using the shape method
    height, width, channels = img.shape

    # Detect the objects in the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net_1.setInput(blob)
    net_2.setInput(blob)
    net_3.setInput(blob)

    outs = net_1.forward(output_layers_1)

    # Show the information on the screen
    class_ids = []
    confidences = []
    boxes = []

    # Loop through all the outputs of the object detected
    for out in outs:
        # Loop through the detections of the output
        for detection in out:
            # Declare the scores, class_id and confidence which will be shown
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected, print the class_id
                print(class_id)

                # Determine the locations of the detected object in the image
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Determine the coordinates of the rectangle to be placed on the screen
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Update values to boxes, confidences and class_ids
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Determine and print the indexes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)

    # Loop through the boxes
    for i in range(len(boxes)):
        if i in indexes:
            # Determine the x, y, w, h of the boxes
            x, y, w, h = boxes[i]

            # Determine the label and color to be used on the image
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # Place the final rectangle and text on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    # Shape detection part----------------------------------------------------------------------------------------------

    # Convert the image into a grayscale image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Determine the threshold and contours of the grayscale image
    _, thresh = cv2.threshold(imgGray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Loop through all the contours in the image for shape detection
    for contour in contours:
        # If the contour area is lower than 150, there was no shape detected
        if cv2.contourArea(contour) < 150:
            cv2.putText(img, "No shape detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            continue

        # Find the approximation lines around the contours to draw
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)

        # Find the approximate x, y values of the contour points
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # Decide what the shape is based on the number of sides
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            #  print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
            else:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 6:
            cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 7:
            cv2.putText(img, "Heptagon", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 8:
            cv2.putText(img, "Octagon", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 9:
            cv2.putText(img, "Nonagon", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        if len(approx) == 10:
            cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))
        else:
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0))

    # Distance from camera to object measurement------------------------------------------------------------------------

    # Define the find_marker to determine the marker point of the image
    def find_marker(image):
        # Convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        # Find the contours in the edged image and keep the largest one
        # Assume that this is the location of the object in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Compute the bounding box of the object's region and return it
        return cv2.minAreaRect(c)

    # Define the distance to camera with the given arguments
    def distance_to_camera(knownWidth, focalLength, perWidth):
        # Compute and return the distance from the marker to the camera
        return (knownWidth * focalLength) / perWidth

    # Initialize the known distance from the camera to the object
    # In this case, is 27 cm
    known_distance = 27.5

    # Initialize the known object width (water pipe), which is about 12 cm
    known_width = 12.5

    # Find the marker in the image, and initialize the focal length
    marker = find_marker(img)
    focalLength = (marker[1][0] * known_distance) / known_width

    # Compute the distance to the marker from the camera
    centimetre = distance_to_camera(known_width, focalLength, marker[1][0])

    # Place the text around the object
    cv2.putText(img, "Distance from camera: %.2fcm" % (centimetre), (img.shape[1] - 400, img.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 0), 1)

    # Object size measurement-------------------------------------------------------------------------------------------

    # Pass the image and grayscale of the original image
    imgOrig = cv2.imread(img_path)
    imgOrig = cv2.resize(imgOrig, None, fx=0.2, fy=0.14)
    imgOrigGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
    ObjectMeasurement.findsize(imgOrig, imgOrigGray)

    # Object distance measurement---------------------------------------------------------------------------------------

    # Pass the image and grayscale of the original image
    DistanceMeasurement.finddistance(imgOrig, imgOrigGray)

    # Image processing output-------------------------------------------------------------------------------------------
    ImageProcessing.processimage(imgOrig, imgOrigGray)

    # Display the image of the detected object
    cv2.imshow("Detected Object", img)

    # If any key is pressed but (esc, s, d, n), then the next image will show
    # If esc key is pressed, then the program will shut down
    # esc = escape all images and close the program
    # s = proceed to the next side measuring
    # d = proceed to the next distance measuring
    # n = proceed to the next captured image
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        continue
    elif key == 27:
        break

cv2.destroyAllWindows()












