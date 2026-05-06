import cv2

from constants import TRAIN_PATH
from dataset import train_detections
from visualization import fetch_images, visualize_using_tracks

images = fetch_images(TRAIN_PATH / "MOT_02" / "img1")
df = train_detections["MOT_02"]

visualize_using_tracks(images, df, "MOT_02.txt")
cv2.destroyAllWindows()
