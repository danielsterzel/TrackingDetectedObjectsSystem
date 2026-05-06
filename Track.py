from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from IoU import calculate_iou
import pandas as pd


@dataclass
class TrackedObject:
    track_id: int
    x: float
    y: float
    w: float
    h: float

    frames_age: int = 0

    time_since_update: int = 0

    def __post_init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1

        self.kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.kf.R *= 10

        self.kf.P *= 1000

        self.kf.Q *= 0.01

        self.kf.x = np.array([[self.x], [self.y], [0], [0]])

    def predict(self):
        self.kf.predict()

        self.x = float(self.kf.x[0, 0])
        self.y = float(self.kf.x[1, 0])

    def update(self, x, y):
        measurement = np.array([[x], [y]])

        self.kf.update(measurement)

        self.x = float(self.kf.x[0, 0])
        self.y = float(self.kf.x[1, 0])

    @property
    def bbox(self):
        return self.x, self.y, self.w, self.h


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 0
        self.iou_threshold = 0.3
        self.max_age = 10

    def create_track(self, detection):

        track = TrackedObject(
            track_id=self.next_track_id,
            x=detection.x,
            y=detection.y,
            w=detection.w,
            h=detection.h,
        )

        self.next_track_id += 1

        self.tracks.append(track)
        print(f"NEW TRACK CREATED " f"id={self.next_track_id}")

    def update_track(self, track: TrackedObject, detection: pd.DataFrame):
        track.update(detection.x, detection.y)
        track.w = detection.w
        track.h = detection.h

        track.frames_age += 1
        track.time_since_update = 0
        print(f"UPDATED Track {track.track_id}")

    def update(self, detections):

        detections_list = list(detections.itertuples())
        print("\n" + "=" * 50)

        print(
            f"UPDATE FRAME | "
            f"tracks={len(self.tracks)} "
            f"detections={len(detections_list)}"
        )

        for track in self.tracks:
            track.predict()

            track.time_since_update += 1

        if len(self.tracks) == 0:

            for detection in detections_list:
                self.create_track(detection)

            return

        cost_matrix = np.zeros((len(self.tracks), len(detections_list)))

        for track_idx, track in enumerate(self.tracks):

            for detection_idx, detection in enumerate(detections_list):
                detection_bbox = (detection.x, detection.y, detection.w, detection.h)

                iou = calculate_iou(track.bbox, detection_bbox)

                cost_matrix[track_idx, detection_idx] = 1 - iou

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        print("\nHungarian matches:")

        matched_tracks = set()

        matched_detections = set()

        for track_idx, detection_idx in zip(track_indices, detection_indices):

            iou = 1 - cost_matrix[track_idx, detection_idx]

            if iou < self.iou_threshold:
                continue

            track = self.tracks[track_idx]

            detection = detections_list[detection_idx]

            self.update_track(track, detection)

            matched_tracks.add(track_idx)

            matched_detections.add(detection_idx)
            print(
                f"Track {track.track_id} "
                f"<-> Detection {detection_idx} "
                f"IoU={iou:.3f}"
            )

        for detection_idx, detection in enumerate(detections_list):

            if detection_idx not in matched_detections:
                self.create_track(detection)

        self.tracks = [
            track for track in self.tracks if track.time_since_update < self.max_age
        ]
