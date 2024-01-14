import cv2
import pyautogui
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a face detection algorithm (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Move the mouse cursor based on the face position
        screen_width, screen_height = pyautogui.size()
        mouse_x = int(screen_width * (x / frame.shape[1]))
        mouse_y = int(screen_height * (y / frame.shape[0]))
        pyautogui.moveTo(mouse_x, mouse_y)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
