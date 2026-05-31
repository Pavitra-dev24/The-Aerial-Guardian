import cv2
import numpy as np


class CMC:
    def __init__(self, num_features=500, quality=0.01, min_dist=3.0):
        self.num_features = num_features
        self.quality = quality
        self.min_dist = min_dist
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        self._prev_gray = None
        self._prev_pts = None

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        identity = np.eye(2, 3, dtype=np.float32)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray)
            return identity

        if self._prev_pts is None or len(self._prev_pts) < 4:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray)
            return identity

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )

        if curr_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray)
            return identity

        valid = status.ravel() == 1
        prev_valid = self._prev_pts[valid].reshape(-1, 2)
        curr_valid = curr_pts[valid].reshape(-1, 2)

        warp = identity
        if len(prev_valid) >= 6:
            M, _ = cv2.estimateAffinePartial2D(
                prev_valid, curr_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if M is not None:
                warp = M.astype(np.float32)

        self._prev_gray = gray
        self._prev_pts = self._detect(gray)
        return warp

    def reset(self):
        self._prev_gray = None
        self._prev_pts = None

    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.num_features,
            qualityLevel=self.quality,
            minDistance=self.min_dist,
            blockSize=3,
        )
