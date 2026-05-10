"""
Camera Motion Compensation (CMC) using sparse optical flow.

Estimates the affine warp between consecutive frames and shifts
Kalman Filter predictions to account for camera movement.
"""

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self):
        self._prev_gray = None
        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
        )
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self):
        self._prev_gray = None

    def compensate(self, frame: np.ndarray, tracks: list) -> None:
        """
        Estimate camera motion and shift all track positions accordingly.
        Modifies track.mean in-place. No-op on first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return

        M = self._estimate_warp(gray)
        if M is not None:
            self._apply(tracks, M)

        self._prev_gray = gray

    def _estimate_warp(self, gray: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(self._prev_gray, **self._feature_params)
        if pts is None or len(pts) < 4:
            return None

        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts, None, **self._lk_params
        )

        good_prev = pts[status[:, 0] == 1]
        good_next = pts_next[status[:, 0] == 1]

        if len(good_prev) < 4:
            return None

        M, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
        return M

    def _apply(self, tracks: list, M: np.ndarray) -> None:
        if not tracks:
            return
        centers = np.array([[t.mean[0], t.mean[1]] for t in tracks], dtype=np.float32)
        centers = centers.reshape(-1, 1, 2)
        warped = cv2.transform(centers, M).reshape(-1, 2)
        for track, (cx, cy) in zip(tracks, warped):
            track.mean[0] = cx
            track.mean[1] = cy
