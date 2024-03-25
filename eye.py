import numpy as np


# Returns EAR given eye landmarks
def eye_aspect_ratio(eye):
    """Used to find the eye aspect ratio on an eye

    Args:
        eye (_type_): The (x,y) coordinate pair array for an eye

    Returns:
        _type_: the eye aspect ratio
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio formula
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear
