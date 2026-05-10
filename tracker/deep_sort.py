"""
DeepSORT tracker.

Per-frame update pipeline
─────────────────────────
1. Predict   — advance every track's Kalman Filter by one step.
2. Extract   — compute ReID embeddings for all detections in this frame.
3. Cascade   — match confirmed tracks to detections via appearance cost
               (NN cosine distance, Mahalanobis gating, Hungarian).
4. IoU       — match remaining confirmed tracks + all tentative tracks
               to remaining detections via IoU cost (positional fallback).
5. Update    — correct matched KF states, update feature galleries.
6. Miss      — mark unmatched tracks; delete old/tentative ones.
7. Init      — create new Tentative tracks from unmatched detections.
"""

import numpy as np

from .kalman_filter import KalmanFilter
from .track import Track, TrackState, Detection
from .reid import ReIDExtractor
from .camera_motion import CameraMotionCompensator
from .matching import (
    matching_cascade,
    iou_cost,
    linear_assignment,
)


class DeepSORT:
    """
    Parameters
    ----------
    max_age        : frames a confirmed track survives without a match
    n_init         : consecutive matches required to confirm a tentative track
    max_cosine_dist: appearance distance threshold for cascade matching
    max_iou_dist   : IoU-cost threshold for the positional fallback step
    min_confidence : detections below this score are discarded
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_dist: float = 0.4,
        max_iou_dist: float = 0.7,
        min_confidence: float = 0.3,
        reid_device: str | None = None,
        use_cmc: bool = True,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_dist = max_cosine_dist
        self.max_iou_dist = max_iou_dist
        self.min_confidence = min_confidence

        self.kf = KalmanFilter()
        self.reid = ReIDExtractor(device=reid_device)
        self.cmc = CameraMotionCompensator() if use_cmc else None

        self.tracks: list[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_image: np.ndarray, raw_detections: np.ndarray) -> list[Track]:
        """
        Process one frame.

        Args:
            frame_image    : BGR uint8 image (H, W, 3)
            raw_detections : (N, 5) array — each row [x, y, w, h, confidence]

        Returns:
            List of confirmed Track objects after this frame.
        """
        # --- filter low-confidence detections ---
        if len(raw_detections) > 0:
            mask = raw_detections[:, 4] >= self.min_confidence
            raw_detections = raw_detections[mask]

        # --- step 1: predict ---
        for track in self.tracks:
            track.predict(self.kf)

        # --- step 1b: camera motion compensation ---
        if self.cmc is not None:
            self.cmc.compensate(frame_image, self.tracks)

        # --- step 2: build Detection objects with ReID embeddings ---
        detections = self._build_detections(frame_image, raw_detections)

        # --- steps 3-4-5-6-7: associate and update ---
        self._match_and_update(detections)

        # --- remove deleted tracks ---
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return [t for t in self.tracks if t.is_confirmed()]

    def reset(self):
        """Reset tracker state between sequences."""
        self.tracks = []
        self._next_id = 1
        if self.cmc is not None:
            self.cmc.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_detections(self, image: np.ndarray, raw: np.ndarray) -> list[Detection]:
        if len(raw) == 0:
            return []

        bboxes = raw[:, :4]
        confs  = raw[:, 4]
        features = self.reid.extract(image, bboxes)

        return [
            Detection(tlwh=bboxes[i], confidence=confs[i], feature=features[i])
            for i in range(len(raw))
        ]

    def _match_and_update(self, detections: list[Detection]) -> None:
        confirmed   = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative   = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_idx = list(range(len(detections)))

        # ── Stage 1: cascade appearance matching (confirmed tracks only) ──
        cascade_matches, unmatched_confirmed, unmatched_dets = matching_cascade(
            kf=self.kf,
            tracks=self.tracks,
            detections=detections,
            confirmed_indices=confirmed,
            det_indices=all_det_idx,
            max_age=self.max_age,
            max_cosine_dist=self.max_cosine_dist,
        )

        # ── Stage 2: IoU fallback (unmatched confirmed + all tentative) ──
        iou_track_idx = unmatched_confirmed + tentative
        iou_matches, unmatched_iou_tracks, unmatched_dets = self._iou_match(
            iou_track_idx, unmatched_dets, detections
        )

        # ── Update matched tracks ──
        for ti, di in cascade_matches + iou_matches:
            self.tracks[ti].update(self.kf, detections[di])

        # ── Mark missed ──
        unmatched_tracks = unmatched_iou_tracks
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # ── Initialise new tracks from unmatched detections ──
        for di in unmatched_dets:
            self._initiate_track(detections[di])

    def _iou_match(self, track_indices, det_indices, detections):
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        cost = iou_cost(self.tracks, detections, track_indices, det_indices)
        matches_idx, unmatched_r, unmatched_c = linear_assignment(
            cost, max_cost=self.max_iou_dist
        )

        matches = [(track_indices[r], det_indices[c]) for r, c in matches_idx]
        unmatched_tracks = [track_indices[r] for r in unmatched_r]
        unmatched_dets   = [det_indices[c]   for c in unmatched_c]
        return matches, unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection: Detection) -> None:
        mean, cov = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean=mean,
                covariance=cov,
                track_id=self._next_id,
                n_init=self.n_init,
                max_age=self.max_age,
                feature=detection.feature,
            )
        )
        self._next_id += 1
