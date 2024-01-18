# Import the necessary libraries
import cv2  # For image and video processing
import pyautogui  # For controlling mouse and keyboard
import numpy as np  # For numerical operations (not explicitly used in this script)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' refers to the default webcam

# Start an infinite loop to process the video frames
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()  # 'ret' is a boolean for frame availability, 'frame' is the captured frame

    # Check if the frame was captured successfully
    if not ret:
        break  # Exit the loop if the frame is not captured

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale simplifies the image processing

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # '1.1' and '4' are parameters for the detection algorithm

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle with a thickness of 2

        # Move the mouse cursor based on the face position
        screen_width, screen_height = pyautogui.size()  # Get screen dimensions
        mouse_x = int(screen_width * (x / frame.shape[1]))  # Calculate the X coordinate for the mouse
        mouse_y = int(screen_height * (y / frame.shape[0]))  # Calculate the Y coordinate for the mouse
        pyautogui.moveTo(mouse_x, mouse_y)  # Move the mouse to the calculated position

    # Display the frame with the rectangle(s)
    cv2.imshow('Frame', frame)  # Display the frame in a window named 'Frame'

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()  # Release the webcam resource
cv2.destroyAllWindows()  # Close all OpenCV windows
