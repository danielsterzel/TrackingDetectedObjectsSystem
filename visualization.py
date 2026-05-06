import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd

from constants import TRAIN_PATH, TEST_PATH
from dataset import train_detections, test_detections

from Track import TrackedObject, Tracker


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

def play_images(images):
    for frame, img in images:

        cv2.imshow("sequence", img)

        key = cv2.waitKey(30)

        if key == 27:
            break

    cv2.destroyAllWindows()

def visualize_using_tracks(
    images,
    df,
    output_file="MOT_02.txt"
):

    tracker = Tracker()

    grouped = df.groupby("frame")

    results = []

    for frame, image in images:

        if frame in grouped.groups:

            sub_df = grouped.get_group(frame)

            tracker.update(sub_df)


        for track in tracker.tracks:

            x1 = int(track.x)
            y1 = int(track.y)

            x2 = int(track.x + track.w)
            y2 = int(track.y + track.h)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"ID: {track.track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            results.append([
                frame,
                track.track_id,
                track.x,
                track.y,
                track.w,
                track.h,
                1,
                -1,
                -1,
                -1
            ])


        cv2.putText(
            image,
            f"FRAME: {frame}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Tracking",
            image
        )

        key = cv2.waitKey(30)

        if key == 27:
            break


    cv2.destroyAllWindows()


    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        header=False,
        index=False
    )

    print(
        f"\nSaved predictions to {output_file}"
    )


images = fetch_images(TRAIN_PATH / "MOT_02" / "img1")
df = train_detections["MOT_02"]

visualize_using_tracks(
    images,
    df,
    "MOT_02.txt"
)
cv2.destroyAllWindows()
