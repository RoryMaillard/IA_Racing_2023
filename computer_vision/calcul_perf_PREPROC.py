import numpy as np
import cv2
import line_profiler


IMAGE_DIR = "datasets\\dataset_drive14\\images"
LABEL_DIR = "datasets\\dataset_drive14"

img=cv2.imread(IMAGE_DIR+"\\140_cam_image_array_.jpg")

TRESH3_MIN = np.array([0, 0, 0], np.uint8)
TRESH3_MAX = np.array([120, 180, 180], np.uint8)
src = np.float32([(10, 69), (150, 73), (118, 60), (35, 60)])
dst = np.float32([(40, 120), (120, 120), (120, 0), (40, 0)])
M = cv2.getPerspectiveTransform(src, dst)

def test_ML():
    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image1=cv2.warpPerspective(image, M, (160, 120))
    image1=255 - cv2.inRange(image1, TRESH3_MIN, TRESH3_MAX)
    image2=255 - cv2.inRange(image, TRESH3_MIN, TRESH3_MAX)
    image3=cv2.Canny(image, 220, 255)
    return cv2.merge((image1, image2, image3))

if __name__ == "__main__":
    profiler = line_profiler.LineProfiler()
    profiler.add_function(test_ML)
    profiler.runcall(test_ML)
    profiler.print_stats()