"""
MOTA evaluation on train sequences using motmetrics.

Usage:
    python evaluate.py --pred-dir output_train_deepsort
"""

import argparse
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from sequence_config import TRAIN_SEQUENCES


def load_gt(seq_path: Path) -> pd.DataFrame:
    gt_path = seq_path / "gt" / "gt.txt"
    cols = ["frame", "id", "x", "y", "w", "h", "eval_flag", "class", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    # keep only evaluatable pedestrians (eval_flag=1, class=1)
    df = df[(df["eval_flag"] == 1) & (df["class"] == 1)]
    return df


def load_pred(pred_path: Path) -> pd.DataFrame:
    cols = ["frame", "id", "x", "y", "w", "h", "c1", "c2", "c3", "c4"]
    return pd.read_csv(pred_path, header=None, names=cols)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_sequence(seq_name: str, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(gt_df["frame"].unique())

    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_ids = gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()

        gt_boxes = gt_frame[["x", "y", "w", "h"]].to_numpy()
        pred_boxes = pred_frame[["x", "y", "w", "h"]].to_numpy()

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            acc.update(gt_ids, pred_ids, [])
            continue

        # distance matrix: 1 - IoU, capped at 1.0 for IoU < 0.5
        dist = np.ones((len(gt_ids), len(pred_ids)))
        for i, gb in enumerate(gt_boxes):
            for j, pb in enumerate(pred_boxes):
                iou = bbox_iou(gb, pb)
                dist[i, j] = 1.0 - iou if iou >= 0.5 else 1.0

        acc.update(gt_ids, pred_ids, dist)

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=Path("output_train_deepsort"))
    args = parser.parse_args()

    accs, names = [], []

    for seq_name, seq_cfg in TRAIN_SEQUENCES.items():
        pred_file = args.pred_dir / f"{seq_name}.txt"
        if not pred_file.exists():
            print(f"  SKIP {seq_name} — no prediction file")
            continue

        gt_df = load_gt(seq_cfg.folder_path)
        pred_df = load_pred(pred_file)

        acc = evaluate_sequence(seq_name, gt_df, pred_df)
        accs.append(acc)
        names.append(seq_name)
        print(f"  {seq_name}: done")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches", "num_misses", "num_false_positives"],
        names=names,
        generate_overall=True,
    )

    summary["mota"] = summary["mota"] * 100
    summary["motp"] = (1 - summary["motp"]) * 100  # convert distance → IoU %
    summary["idf1"] = summary["idf1"] * 100

    print("\n" + mm.io.render_summary(
        summary,
        formatters={"mota": "{:.1f}%".format, "motp": "{:.1f}%".format, "idf1": "{:.1f}%".format},
        namemap={
            "num_frames": "Frames",
            "mota": "MOTA",
            "motp": "MOTP",
            "idf1": "IDF1",
            "num_switches": "ID Sw",
            "num_misses": "FN",
            "num_false_positives": "FP",
        },
    ))


if __name__ == "__main__":
    main()
