"""
Track and Detection data classes.
"""

from enum import Enum, auto
import numpy as np

_EMA_ALPHA = 0.9  # smoothing factor for appearance feature EMA


class TrackState(Enum):
    Tentative = auto()   # newly created, not yet confirmed
    Confirmed = auto()   # confirmed, reported in output
    Deleted   = auto()   # to be removed from tracker


class Detection:
    """Single detection in one frame."""

    def __init__(self, tlwh: np.ndarray, confidence: float, feature: np.ndarray):
        self.tlwh = np.asarray(tlwh, dtype=float)     # [x, y, w, h]
        self.confidence = confidence
        self.feature = feature                         # ReID embedding

    def to_xyah(self) -> np.ndarray:
        """Convert to Kalman Filter measurement [cx, cy, a, h]."""
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2, w / h, h])

    def to_tlbr(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h])


class Track:
    """
    A single tracked object.

    Lifecycle:
      Tentative → Confirmed (after min_hits consecutive matches)
      Confirmed → Deleted   (after max_age frames without a match)
      Tentative → Deleted   (if not matched in first frame)
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: np.ndarray | None = None,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id

        self.hits = 1                        # consecutive matched frames
        self.age = 1                         # total frames since creation
        self.time_since_update = 0           # frames since last match

        self.state = TrackState.Tentative

        # Gallery: stores recent appearance embeddings for this track
        self.features: list[np.ndarray] = []
        self.smooth_feat: np.ndarray | None = None
        if feature is not None:
            self.features.append(feature)
            self.smooth_feat = feature.copy()

        self._n_init = n_init
        self._max_age = max_age

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def tlwh(self) -> np.ndarray:
        """Current bbox estimate as [x, y, w, h]."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def tlbr(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h])

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection: Detection) -> None:
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )

        feat = detection.feature

        # EMA smooth feature — stable representation across appearance changes
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            updated = _EMA_ALPHA * self.smooth_feat + (1 - _EMA_ALPHA) * feat
            norm = np.linalg.norm(updated)
            self.smooth_feat = updated / (norm + 1e-6)

        # Keep a capped gallery of recent raw embeddings
        self.features.append(feat)
        if len(self.features) > 50:
            self.features = self.features[-50:]

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
