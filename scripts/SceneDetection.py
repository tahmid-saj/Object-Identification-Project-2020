#Work done by: Tahmid Sajin
#Email for contact: tahmid.sajin@ryerson.ca

#Program description (Please read carefully):
#This program is the scene detection part of the waste water monitoring system.
#The main purpose is to detect a scene change detection, in which an object appears within the pi camera's vision.
#If detected, the program will pass the detected object's frame and greyscale frame to the AWS S3 bucket assigned.
#The program will also save the detected frames into the project file for further manual examination if required.
#The detection will be determined using contours that are calculated within each frame.
#The program will loop through each frame, comparing the current frame with the previous frame to determine a change.
#If the change within the two consecutive frames show a large difference past the threshold, it will assume detection.
#Further mse values will also be calculated, but this will slow the pi camera, and is advised in not using for fast responses.
#The program can be run using the "run" button without passing any arguments.
#The program has a memory for the following amount of frames (10000) or time indicated in the code (please see below).
#Please change the background.png image in the project folder as desired for the chosen location in the waste water system.
#This is only the fundamental application of scene detection and can be further improved or tailored to the specific task if required.
#Please email or contact for any information or inquiry.

#Import all the relevant libraries, packages, methods etc
#from skimage.metrics import structural_similarity as ssim #this has been commented.
#There is no need to uncomment unless ssim is required.
#Please email if there is a need for ssim for a particular situation. ssim will make the program much slower.
#Only uncomment if ssim (structural similarity index) is required. This will lower the response time of the pi camera.
import imutils #Will not need this if not using any imutils methods
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import datetime
import boto3
from botocore.client import Config

#Initialize the relevant variables, flags and counters to be used in the code
#mse_val_avg = 0 #If using ssim, then uncomment
num_frames = 0

#Declare the file to be opened, to store the ssim, and truncate it
#mse_val_file = open("MSE_values.txt", 'w') #If using ssim, then uncomment
#mse_val_file.truncate(0) #If using ssim, then uncomment

#Initialize the frame dimensions (They will be set
#as soon as we read the first frame from the video)
W = None
H = None

#Initialize the AWS access and bucket variables
#Note: If using a different AWS account, this will need to be changed.
#Follow the report for more information to setup an AWS S3 bucket and configure it to
#allow python to send images via the cloud.
ACCESS_KEY_ID = '' #Enter the access key id
ACCESS_SECRET_KEY = '' #Enter the access secret key
BUCKET_NAME = '' #Enter the bucket name

#Declare the AWS S3 bucket identification
s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

#Initialize the video stream and allow the camera sensor to warmup
print("Warming up camera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(5.0)

#Read the first and second frame of the pi camera
frame1 = vs.read()
frame2 = vs.read()

#Loop over the frames of the stream
while True:
    #The loop will grab both the previous frame from the stream and current

    #Quit if there was a problem grabbing a frame
    if frame1 is None and frame2 is None:
        break

    #Resize the frame
    frame1 = imutils.resize(frame1, width=1000)
    frame2 = imutils.resize(frame2, width=1000)

    #If the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame1.shape[:2]

    #Increment the number of frames counter
    num_frames = num_frames + 1

    #Find the difference, grayscale, blur, threshold and contours of the image in that order
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Find the mse of two consecutive frames. Uncomment the following two lines if using ssim
    #def mse(x, y):
    #   return np.linalg.norm(x - y)

    #Find the mse_val of two consecutive frames. Pass the arguments to mse. Uncomment the following line if using ssim
    #mse_val = mse(frame1, frame2)

    #Find the mse_val_avg of the two consecutive frames. Uncomment the following four lines if using ssim
    #for i in range (num_frames):
    #	mse_val_avg = mse_val + mse_val_avg

    #mse_val_avg = mse_val_avg / num_frames

    #Print the mse_val and mse_val_avg values and write them to the file. Uncomment is using ssim
    #print("Frame number: %d     MSE value: %d    MSE average: %d" %(num_frames, mse_val, mse_val_avg))
    #mse_val_file.write("%d,%d,%d\n" %(num_frames, mse_val, mse_val_avg))

    #Find the contours of the current frame
    for contour in contours:
        #Find the x, y, w, h values of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        #Find the contour area and see if its less than 500 to be considered a significant change in scene
        if (cv2.contourArea(contour) < 500):
            continue
        #Else if the contour area is under 500. Place a rectangle on the contour and update the text of the frame
        elif (cv2.contourArea(contour) >= 500):
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 255), 1) #----------------------Comment this if there is no need to place a rectangle
            cv2.putText(frame1, "Object {}".format("Detected"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

            #Find the current date and time
            datet = str(datetime.datetime.now())

            #Display the current date and time into the frame
            frame1 = cv2.putText(frame1, datet, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

            #Equate the frame number to the text of the image to be saved
            text = "frame" + str(num_frames) + ".png"
            text_background = "framegreyscale" + str(num_frames) + ".png"

            #Save the detected image and send it via AWS into the S3 bucket to be processed
            cv2.imwrite(text, frame1)
            img = open(text, 'rb')
            s3.Bucket(BUCKET_NAME).put_object(Key=text, Body=img)

            #Find the greyscale background subtraction image of the detected object
            #background = cv2.imread('background.png')
            #diff_background = cv2.absdiff(frame1, background)
            #gray_background = cv2.cvtColor(diff_background, cv2.COLOR_BGR2GRAY)

            #Save the greyscale background subtraction image and send it into the AWS S3 bucket
            #cv2.imwrite(text_background, gray_background)
            #img2 = open(text_background, 'rb')
            #s3.Bucket(BUCKET_NAME).put_object(Key=text_background, Body=img2)

    #Show the stream
    cv2.imshow("Stream", frame1)

    #Equate frame2 to frame 1 and read the next frame for frame2
    frame1 = frame2
    frame2 = vs.read()

    #Close the program
    if cv2.waitKey(40) == 27:
        #mse_val_file.close() #Close the mse_val_file
        break

#Cleanup the camera and close any open windows
cv2.destroyAllWindows()
vs.stop()











