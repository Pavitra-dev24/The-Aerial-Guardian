import numpy as np
from .kalman import KalmanFilter
from .matching import iou_matrix, linear_assignment


class Track:
    _count = 0

    @classmethod
    def _next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset_count(cls):
        cls._count = 0

    def __init__(self, det_xyxy, score, n_init=3, max_age=30):
        self.track_id = self._next_id()
        self.hits = 1
        self.time_since_update = 0
        self.score = score
        self._n_init = n_init
        self._confirmed = False
        self._kf = KalmanFilter()
        z = self._to_z(det_xyxy)
        self.mean, self.cov = self._kf.initiate(z)
        self.tail = []

    @staticmethod
    def _to_z(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        cx = box[0] + w / 2.0
        cy = box[1] + h / 2.0
        return np.array([cx, cy, w / (h + 1e-6), h], dtype=float)

    def predict(self):
        self.mean, self.cov = self._kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, det_xyxy, score):
        z = self._to_z(det_xyxy)
        self.mean, self.cov = self._kf.update(self.mean, self.cov, z)
        self.hits += 1
        self.time_since_update = 0
        self.score = score
        if self.hits >= self._n_init:
            self._confirmed = True

    def apply_warp(self, warp):
        self.mean, self.cov = self._kf.apply_warp(self.mean, self.cov, warp)

    def record_position(self, max_tail=50):
        self.tail.append((float(self.mean[0]), float(self.mean[1])))
        if len(self.tail) > max_tail:
            self.tail.pop(0)

    def is_confirmed(self):
        return self._confirmed

    @property
    def xyxy(self):
        cx, cy, ar, h = self.mean[:4]
        w = ar * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class ByteTracker:
    """
    ByteTrack with Camera Motion Compensation.

    Two-stage association:
    - Stage 1: high-confidence detections vs all active tracks (IoU)
    - Stage 2: low-confidence detections vs remaining unmatched tracks (IoU)

    CMC pre-warps Kalman predictions by the estimated inter-frame affine
    transform before IoU matching, compensating for drone ego-motion and
    preventing spurious ID switches caused by global camera movement.
    """

    def __init__(self, track_thresh=0.45, match_thresh=0.5,
                 second_thresh=0.5, max_age=30, n_init=3):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_thresh = second_thresh
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []

    def reset(self):
        self.tracks = []
        Track.reset_count()

    def update(self, dets, scores, warp=None):
        dets = np.asarray(dets, dtype=float).reshape(-1, 4) if len(dets) > 0 else np.empty((0, 4))
        scores = np.asarray(scores, dtype=float).reshape(-1) if len(scores) > 0 else np.empty(0)

        for t in self.tracks:
            t.predict()

        if warp is not None:
            for t in self.tracks:
                t.apply_warp(warp)

        high_mask = scores >= self.track_thresh
        high_dets = dets[high_mask]
        high_scores = scores[high_mask]
        low_dets = dets[~high_mask]
        low_scores = scores[~high_mask]

        m1, ut1, ud1 = self._match(self.tracks, high_dets, self.match_thresh)
        for ti, di in m1:
            self.tracks[ti].update(high_dets[di], high_scores[di])

        rem_tracks = [self.tracks[i] for i in ut1]
        m2, _, _ = self._match(rem_tracks, low_dets, self.second_thresh)
        for ti, di in m2:
            rem_tracks[ti].update(low_dets[di], low_scores[di])

        for i in ud1:
            self.tracks.append(
                Track(high_dets[i], high_scores[i], self.n_init, self.max_age)
            )

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        for t in self.tracks:
            t.record_position()

        return [t for t in self.tracks if t.is_confirmed()]

    def _match(self, tracks, dets, iou_threshold):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        track_boxes = np.array([t.xyxy for t in tracks])
        cost = 1.0 - iou_matrix(track_boxes, dets)
        return linear_assignment(cost, 1.0 - iou_threshold)
