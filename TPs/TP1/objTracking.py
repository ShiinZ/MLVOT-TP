from Detector import detect
from KalmanFilter import KalmanFilter
import cv2
import numpy as np

def kalman(frame):
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    centers = detect(frame)
    if (len(centers) > 0):
        # Predict
        (x, y) = KF.predict()
        AugmentSize = 16
        cv2.rectangle(frame, (int(x - AugmentSize), int(y - AugmentSize)), (int(x + AugmentSize), int(y + AugmentSize)), (255, 0, 0), 2)
        
        # Update
        (x1, y1) = KF.update(centers[0])
        cv2.rectangle(frame, (int(x1 - AugmentSize), int(y1 - AugmentSize)), (int(x1 + AugmentSize), int(y1 + AugmentSize)), (0, 0, 255), 2)
    return 
