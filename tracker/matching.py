"""
Association utilities for DeepSORT.

Two-stage matching strategy:
  1. Cascade matching  – appearance (cosine) + Mahalanobis gating  → confirmed tracks
  2. IoU matching      – positional fallback for remaining tracks/detections
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between every pair (a, b).
    Both arrays are [N, 4] or [M, 4] in [x, y, w, h] format.
    Returns (N, M) IoU matrix.
    """
    ax1 = boxes_a[:, 0]
    ay1 = boxes_a[:, 1]
    ax2 = boxes_a[:, 0] + boxes_a[:, 2]
    ay2 = boxes_a[:, 1] + boxes_a[:, 3]

    bx1 = boxes_b[:, 0]
    by1 = boxes_b[:, 1]
    bx2 = boxes_b[:, 0] + boxes_b[:, 2]
    by2 = boxes_b[:, 1] + boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


def iou_cost(tracks, detections, track_indices, det_indices):
    """
    Cost matrix (1 - IoU) for the given subset of tracks / detections.
    Uses the Kalman-predicted bbox for each track.
    """
    t_boxes = np.array([tracks[i].tlwh for i in track_indices])
    d_boxes = np.array([detections[j].tlwh for j in det_indices])
    return 1.0 - box_iou_batch(t_boxes, d_boxes)


# ---------------------------------------------------------------------------
# Appearance (cosine) helpers
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance — (N, D) × (M, D) → (N, M)."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - a @ b.T


def nn_cosine_cost(tracks, detections, track_indices, det_indices):
    """
    Nearest-neighbour cosine cost: for each track, take the *minimum*
    cosine distance across its entire feature gallery.
    This is more robust than using only the latest embedding.
    """
    det_features = np.array([detections[j].feature for j in det_indices])
    cost = np.zeros((len(track_indices), len(det_indices)))

    for row, ti in enumerate(track_indices):
        gallery = np.array(tracks[ti].features)         # (K, D)
        dists = cosine_distance(gallery, det_features)  # (K, M)
        cost[row] = dists.min(axis=0)

    return cost


# ---------------------------------------------------------------------------
# Mahalanobis gating
# ---------------------------------------------------------------------------

# Chi-squared threshold for 4-dim measurements at p = 0.95
MAHALANOBIS_THRESH = 9.4877


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, det_indices):
    """
    Set cost to a large sentinel for pairs where the Mahalanobis distance
    (predicted KF position → detection) exceeds the chi-squared threshold.
    This prunes geometrically impossible associations before Hungarian.
    """
    measurements = np.array([detections[j].to_xyah() for j in det_indices])

    for row, ti in enumerate(track_indices):
        maha = kf.gating_distance(
            tracks[ti].mean, tracks[ti].covariance, measurements
        )
        cost_matrix[row, maha > MAHALANOBIS_THRESH] = 1e5

    return cost_matrix


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def linear_assignment(cost_matrix: np.ndarray, max_cost: float):
    """
    Run Hungarian algorithm and split result into:
      matches         – list of (track_index, det_index) tuples
      unmatched_rows  – row indices with no valid match
      unmatched_cols  – col indices with no valid match
    """
    if cost_matrix.size == 0:
        rows = list(range(cost_matrix.shape[0]))
        cols = list(range(cost_matrix.shape[1]))
        return [], rows, cols

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = set()
    matches, unmatched_rows, unmatched_cols = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((r, c))
            matched.add((r, c))

    for r in range(cost_matrix.shape[0]):
        if not any(r == m[0] for m in matches):
            unmatched_rows.append(r)

    for c in range(cost_matrix.shape[1]):
        if not any(c == m[1] for m in matches):
            unmatched_cols.append(c)

    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Cascade matching  (DeepSORT core)
# ---------------------------------------------------------------------------

def matching_cascade(
    kf,
    tracks,
    detections,
    confirmed_indices,
    det_indices,
    max_age: int,
    max_cosine_dist: float = 0.4,
):
    """
    Match confirmed tracks to detections, prioritising recently-seen tracks.

    Iterates age levels 1 … max_age:
      - At each level, considers only tracks unseen for exactly that many frames.
      - Uses appearance cost (NN cosine) gated by Mahalanobis distance.
      - Unmatched detections carry over to the next level.

    Returns (matches, unmatched_confirmed, unmatched_dets)
    where indices are positions in the original `tracks` / `detections` lists.
    """
    unmatched_dets = list(det_indices)
    all_matches = []

    for age in range(1, max_age + 1):
        level_tracks = [i for i in confirmed_indices if tracks[i].time_since_update == age]
        if not level_tracks or not unmatched_dets:
            continue

        cost = nn_cosine_cost(tracks, detections, level_tracks, unmatched_dets)
        cost = gate_cost_matrix(kf, cost, tracks, detections, level_tracks, unmatched_dets)

        level_matches, _, remaining_dets = linear_assignment(cost, max_cost=max_cosine_dist)

        for r, c in level_matches:
            all_matches.append((level_tracks[r], unmatched_dets[c]))

        unmatched_dets = [unmatched_dets[c] for c in remaining_dets]

    matched_track_ids = {ti for ti, _ in all_matches}
    unmatched_confirmed = [i for i in confirmed_indices if i not in matched_track_ids]

    return all_matches, unmatched_confirmed, unmatched_dets
