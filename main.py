import cv2
import dlib
import numpy as np
import pyautogui

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "C:\\Users\\shape_predictor_68_face_landmarks (1).dat"
predictor = dlib.shape_predictor(predictor_path)

# Safety margin to prevent the mouse from hitting the screen edge
safe_margin = 100

def get_head_pose(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]  # Use the first detected face
        landmarks = predictor(gray, face)
        
        # Define image points based on facial landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype="double")

        # Pre-defined model points
        model_points = np.array([
            (0.0, 0.0, 0.0),              # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ])
        
        # Camera internals
        size = image.shape
        focal_length = size[1]
        center = (size[1]//2, size[0]//2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4,1))
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        return rotation_vector, translation_vector
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rotation_vector, translation_vector = get_head_pose(frame, detector, predictor)
        if translation_vector is not None:
            # Simplified mapping from translation_vector to screen coordinates
            x_translation, y_translation = translation_vector[0][0], translation_vector[1][0]
            screen_x = np.interp(x_translation, [-100, 100], [screen_width - safe_margin, safe_margin])
            screen_y = np.interp(y_translation, [-100, 100], [safe_margin, screen_height - safe_margin])
            
            # Move the mouse cursor with smoothing
            current_mouse_x, current_mouse_y = pyautogui.position()
            target_x = (screen_x + current_mouse_x) / 2
            target_y = (screen_y + current_mouse_y) / 2
            pyautogui.moveTo(target_x, target_y, duration=0.1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
