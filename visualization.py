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
def view_detections(images, df: pd.DataFrame):

    grouped = df.groupby("frame")

    for frame, img in images:

        if frame in grouped.groups:

            sub_df = grouped.get_group(frame)

            for detection in sub_df.itertuples():

                x1 = int(detection.x)
                y1 = int(detection.y)
                w = int(detection.w)
                h = int(detection.h)

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x1 + w, y1 + h),
                    (0, 255, 0),
                    2
                )

        cv2.imshow("detections", img)

        key = cv2.waitKey(30)

        if key == 27:
            break

    cv2.destroyAllWindows()

images = fetch_images(TRAIN_PATH / "MOT_02" / "img1")
df = train_detections["MOT_02"]

view_detections(images, df)

def play_images(images):
    for frame, img in images:

        cv2.imshow("sequence", img)

        key = cv2.waitKey(30)

        if key == 27:
            break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
