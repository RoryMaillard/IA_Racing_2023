import cv2
import numpy as np

class Preprocessor:
    def __init__(self, config=None):
        self.TRESH3_MIN = np.array([0, 0, 0], np.uint8)
        self.TRESH3_MAX = np.array([120, 180, 180], np.uint8)
        self.src = np.float32([(10, 69), (150, 73), (118, 60), (35, 60)])
        self.dst = np.float32([(40, 120), (120, 120), (120, 0), (40, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

    def run(self, imagepre):
        try:
            image=cv2.cvtColor(imagepre, cv2.COLOR_BGR2RGB)
            image1=cv2.warpPerspective(image, self.M, (160, 120))
            image1=255 - cv2.inRange(image1, self.TRESH3_MIN, self.TRESH3_MAX)
            image2=255 - cv2.inRange(image, self.TRESH3_MIN, self.TRESH3_MAX)
            image3=cv2.Canny(image, 220, 255)
            return cv2.merge((image1, image2, image3))
        except:
            return None
    
    def shutdown(self):
        pass