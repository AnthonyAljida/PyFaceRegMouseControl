import cv2
import dlib
import numpy as np
import pyautogui
import imutils

# All constants come from constants.py
from constants import *
from imutils import face_utils
from eye import eye_aspect_ratio
from mouth import mouth_aspect_ratio
from nose import direction
from voice import start_voice_recognition

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


# Thresholds and consecutive frame length for triggering the mouse action.
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 12
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 12
WINK_AR_DIFF_THRESH = 0.03
WINK_CONSECUTIVE_FRAMES = 5

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


start_voice_recognition()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # If no camera, program is unable to work
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=WIDTH_FOR_CAMERA, height=HEIGHT_FOR_CAMERA)
    # Used for converting image to an easy color to process, grey is best for images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Grab all the faces in the camera
    faces = detector(gray, 0)
    # Loop over the face detections and grab the first face
    if len(faces) > 0:
        face = faces[0]

    # Guard statement for no face
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Take the 68 landmarks processed from the face and make them into
    # a more easy to understand format for python, (I.E. an array is easier to use then the raw landmark format)
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    # This link is all the documentation needed to understand this part
    # https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/helpers.py
    mouth = landmarks[mStart:mEnd]
    leftEye = landmarks[lStart:lEnd]
    rightEye = landmarks[rStart:rEnd]
    nose = landmarks[nStart:nEnd]

    # Swap because the frame is flipped
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Average the mouth aspect ratio together for both eyes
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    for x, y in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if diff_ear > WINK_AR_DIFF_THRESH:

        if leftEAR < rightEAR:
            if leftEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    pyautogui.click(button="left")

                    WINK_COUNTER = 0

        elif leftEAR > rightEAR:
            if rightEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    pyautogui.click(button="right")

                    WINK_COUNTER = 0
        else:
            WINK_COUNTER = 0
    else:
        if ear <= EYE_AR_THRESH:
            EYE_COUNTER += 1

            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                SCROLL_MODE = not SCROLL_MODE
                # INPUT_MODE = not INPUT_MODE
                EYE_COUNTER = 0

                # nose point to draw a bounding box around it

        else:
            EYE_COUNTER = 0
            WINK_COUNTER = 0

    if mar > MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1

        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
            # if the alarm is not on, turn it on
            INPUT_MODE = not INPUT_MODE
            # SCROLL_MODE = not SCROLL_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point

    else:
        MOUTH_COUNTER = 0

    if INPUT_MODE:
        cv2.putText(
            frame,
            "READING INPUT!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED_COLOR,
            2,
        )
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(
            frame,
            dir.upper(),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED_COLOR,
            2,
        )
        drag = 14
        if dir == "right":
            pyautogui.moveRel(drag, 0)
        elif dir == "left":
            pyautogui.moveRel(-drag, 0)
        elif dir == "up":
            if SCROLL_MODE:
                pyautogui.scroll(40)
            else:
                pyautogui.moveRel(0, -drag)
        elif dir == "down":
            if SCROLL_MODE:
                pyautogui.scroll(-40)
            else:
                pyautogui.moveRel(0, drag)

    if SCROLL_MODE:
        cv2.putText(
            frame,
            "SCROLL MODE IS ON!",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED_COLOR,
            2,
        )

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `Esc` key was pressed, end the program
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
