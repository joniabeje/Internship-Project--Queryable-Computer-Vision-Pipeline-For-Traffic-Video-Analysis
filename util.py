import numpy as np
import cv2

def get_limits(color_name):
    if color_name == 'Red':
        # Adjusted Red color ranges to avoid overlap with Yellow
        lowerLimit1 = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit1 = np.array([10, 255, 255], dtype=np.uint8)
        lowerLimit2 = np.array([165, 100, 100], dtype=np.uint8)
        upperLimit2 = np.array([180, 255, 255], dtype=np.uint8)
        return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)
    elif color_name == 'Yellow':
        # Yellow color ranges
        lowerLimit = np.array([10, 100, 100], dtype=np.uint8)  # Adjusted lower limit
        upperLimit = np.array([35, 255, 255], dtype=np.uint8)  # Adjusted upper limit
        return lowerLimit, upperLimit
    elif color_name == 'Green':
        # Green color ranges
        lowerLimit = np.array([35, 50, 50], dtype=np.uint8)  # Adjusted lower limit
        upperLimit = np.array([90, 255, 255], dtype=np.uint8)  # Adjusted upper limit
        return lowerLimit, upperLimit

def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply a series of erosions and dilations to remove small noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    return mask
