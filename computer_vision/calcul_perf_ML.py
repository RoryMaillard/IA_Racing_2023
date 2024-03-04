import numpy as np
import cv2
import line_profiler

from keras.saving import load_model

IMAGE_DIR = "datasets\\dataset_drive14\\images"
LABEL_DIR = "datasets\\dataset_drive14"

model= load_model("models/model.h5")

img=cv2.imread(IMAGE_DIR+"\\140_cam_image_array_.jpg")


def test_ML():
    TRESH3_MIN = np.array([0, 0, 0],np.uint8)
    TRESH3_MAX = np.array([255, 120, 255],np.uint8)
    X=255-cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), TRESH3_MIN, TRESH3_MAX,)[50:][:]
    X=X/255

    X = X.reshape(-1, 70, 160, 1)
    
    Y=model(X, training=False)
    print(Y)

if __name__ == "__main__":
    profiler = line_profiler.LineProfiler()
    profiler.add_function(test_ML)
    profiler.runcall(test_ML)
    profiler.print_stats()