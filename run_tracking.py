"""
Main tracking script.

For each sequence (train or test), runs DeepSORT and writes results
to the required submission format:
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sequence_config import TRAIN_SEQUENCES, TEST_SEQUENCES, SequenceConfig
from tracker import DeepSORT

# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_detections(seq: SequenceConfig) -> pd.DataFrame:
    det_path = seq.folder_path / "det" / "det.txt"
    cols = ["frame", "id", "x", "y", "w", "h", "confidence"]
    return pd.read_csv(det_path, header=None, names=cols)


def frame_image_path(seq: SequenceConfig, frame_num: int) -> Path:
    return seq.folder_path / seq.imDir / f"{frame_num:06d}{seq.imExt}"


def run_sequence(seq: SequenceConfig, tracker: DeepSORT, output_path: Path) -> None:
    det_df = load_detections(seq)
    grouped = det_df.groupby("frame")

    lines = []

    for frame_num in tqdm(range(1, seq.seqLength + 1), desc=seq.name, ncols=80):
        img_path = frame_image_path(seq, frame_num)
        image = cv2.imread(str(img_path))

        if image is None:
            # Missing frame — still need to advance tracker (predict only)
            image = np.zeros((seq.imHeight, seq.imWidth, 3), dtype=np.uint8)

        # Build raw detections array [x, y, w, h, confidence]
        if frame_num in grouped.groups:
            sub = grouped.get_group(frame_num)
            raw = sub[["x", "y", "w", "h", "confidence"]].to_numpy(dtype=float)
        else:
            raw = np.empty((0, 5), dtype=float)

        confirmed_tracks = tracker.update(image, raw)

        for track in confirmed_tracks:
            x, y, w, h = track.tlwh
            lines.append(
                f"{frame_num},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  → wrote {len(lines)} lines to {output_path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run DeepSORT on MOT sequences")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which dataset split to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for result .txt files",
    )
    # Tracker hyper-parameters
    parser.add_argument("--max-age",         type=int,   default=30)
    parser.add_argument("--n-init",          type=int,   default=3)
    parser.add_argument("--max-cosine-dist", type=float, default=0.4)
    parser.add_argument("--max-iou-dist",    type=float, default=0.7)
    parser.add_argument("--min-confidence",  type=float, default=0.3)
    args = parser.parse_args()

    tracker_kwargs = dict(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_dist=args.max_cosine_dist,
        max_iou_dist=args.max_iou_dist,
        min_confidence=args.min_confidence,
    )

    sequences_to_run = {}
    if args.split in ("train", "both"):
        sequences_to_run.update(TRAIN_SEQUENCES)
    if args.split in ("test", "both"):
        sequences_to_run.update(TEST_SEQUENCES)

    print(f"Sequences: {list(sequences_to_run.keys())}")
    print(f"Tracker params: {tracker_kwargs}\n")

    tracker = DeepSORT(**tracker_kwargs)

    for name, seq in sequences_to_run.items():
        tracker.reset()
        out_file = args.output_dir / f"{name}.txt"
        run_sequence(seq, tracker, out_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
