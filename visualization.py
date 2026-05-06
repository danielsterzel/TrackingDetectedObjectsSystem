import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd

from constants import TRAIN_PATH, TEST_PATH
from sequence_config import TRAIN_SEQUENCES, TEST_SEQUENCES
from dataset import train_detections, test_detections


def fetch_images(sequence_images: Path):

    for image_path in sorted(sequence_images.iterdir()):

        if not image_path.is_file() or image_path.suffix.lower() not in [
            ".jpg",
            ".png",
        ]:
            continue

        image = cv2.imread(image_path)
        frame_number = int(image_path.stem)

        if image is None:
            continue

        yield frame_number, image


def sample_and_show_images(data: tuple, num_samples=10):

    for idx, (frame, img) in enumerate(data):

        cv2.imshow(str(frame), img)

        cv2.waitKey(0)

        if idx == num_samples:
            break


#
def view_detections(images, df: pd.DataFrame, num_samples=10):

    for idx, (frame, img) in enumerate(images):

        if idx > num_samples:
            break

        sub_df = df[df["frame"] == frame]

        for _, detection in sub_df.iterrows():

            x1, y1, w, h = (
                int(detection["x"]),
                int(detection["y"]),
                int(detection["w"]),
                int(detection["h"]),
            )
            x2 = x1 + w
            y2 = y1 + h

            cv2.rectangle(img,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)
            cv2.imshow(str(frame), img)
            cv2.waitKey(0)

images = fetch_images(TRAIN_PATH / "MOT_02" / "img1")
df = train_detections["MOT_02"]

view_detections(images, df)

cv2.destroyAllWindows()