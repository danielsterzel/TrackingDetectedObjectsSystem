"""
Visualization of DeepSORT tracking.

Usage:
    python visualize_deepsort.py --sequence MOT_02
    python visualize_deepsort.py --sequence MOT_02 --save output.mp4
    python visualize_deepsort.py --sequence MOT_01 --split test
"""

import argparse
import random

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sequence_config import TRAIN_SEQUENCES, TEST_SEQUENCES
from tracker import DeepSORT


def get_color(track_id: int) -> tuple:
    random.seed(track_id)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_tracks(image: np.ndarray, tracks, frame_num: int) -> np.ndarray:
    for track in tracks:
        x, y, w, h = track.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        color = get_color(track.track_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"ID {track.track_id}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.putText(
        image,
        f"Frame: {frame_num}  Tracks: {len(tracks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="MOT_02")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--save", type=Path, default=None, help="Save to video file")
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()

    sequences = TRAIN_SEQUENCES if args.split == "train" else TEST_SEQUENCES
    if args.sequence not in sequences:
        print(f"Sequence '{args.sequence}' not found. Available: {list(sequences.keys())}")
        return

    seq = sequences[args.sequence]

    import pandas as pd
    det_path = seq.folder_path / "det" / "det.txt"
    det_df = pd.read_csv(det_path, header=None, names=["frame", "id", "x", "y", "w", "h", "confidence"])
    grouped = det_df.groupby("frame")

    tracker = DeepSORT(max_age=args.max_age, n_init=args.n_init, min_confidence=args.min_confidence)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, seq.frameRate, (seq.imWidth, seq.imHeight))

    print(f"Sequence: {seq.name}  |  {seq.seqLength} frames  |  {seq.imWidth}x{seq.imHeight}")
    print("Press ESC to quit, SPACE to pause.")

    paused = False

    for frame_num in tqdm(range(1, seq.seqLength + 1), ncols=80):
        img_path = seq.folder_path / seq.imDir / f"{frame_num:06d}{seq.imExt}"
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((seq.imHeight, seq.imWidth, 3), dtype=np.uint8)

        if frame_num in grouped.groups:
            sub = grouped.get_group(frame_num)
            raw = sub[["x", "y", "w", "h", "confidence"]].to_numpy(dtype=float)
        else:
            raw = np.empty((0, 5), dtype=float)

        confirmed = tracker.update(image, raw)
        vis = draw_tracks(image.copy(), confirmed, frame_num)

        if writer:
            writer.write(vis)

        cv2.imshow(f"DeepSORT — {seq.name}", vis)

        delay = 0 if paused else max(1, int(1000 / seq.frameRate))
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:   # ESC
            break
        if key == 32:   # SPACE
            paused = not paused

    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
