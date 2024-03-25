import numpy as np


# Return direction given the nose and anchor points.
def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return "right"
    elif nx < x - multiple * w:
        return "left"

    if ny > y + multiple * h:
        return "down"
    elif ny < y - multiple * h:
        return "up"

    return "none"
