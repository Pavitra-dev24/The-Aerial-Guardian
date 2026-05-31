import numpy as np


class KalmanFilter:
    ndim = 4

    def __init__(self):
        n = self.ndim
        self.F = np.eye(2 * n)
        for i in range(n):
            self.F[i, n + i] = 1.0
        self.H = np.eye(n, 2 * n)
        self._sp = 1.0 / 20
        self._sv = 1.0 / 160

    def _Q(self, h):
        s = [
            self._sp * h, self._sp * h, 1e-2, self._sp * h,
            self._sv * h, self._sv * h, 1e-5, self._sv * h,
        ]
        return np.diag(np.square(s))

    def _R(self, h):
        s = [self._sp * h, self._sp * h, 1e-1, self._sp * h]
        return np.diag(np.square(s))

    def initiate(self, z):
        x = np.r_[z, np.zeros(self.ndim)]
        h = z[3]
        sp, sv = self._sp, self._sv
        P = np.diag(np.square([
            2 * sp * h, 2 * sp * h, 1e-2, 2 * sp * h,
            10 * sv * h, 10 * sv * h, 1e-5, 10 * sv * h,
        ]))
        return x, P

    def predict(self, x, P):
        Q = self._Q(max(x[3], 1.0))
        x = self.F @ x
        P = self.F @ P @ self.F.T + Q
        return x, P

    def update(self, x, P, z):
        R = self._R(max(x[3], 1.0))
        S = self.H @ P @ self.H.T + R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ (z - self.H @ x)
        P = (np.eye(len(x)) - K @ self.H) @ P
        return x, P

    def apply_warp(self, x, P, warp):
        R2 = warp[:2, :2]
        t = warp[:2, 2]
        x = x.copy()
        P = P.copy()
        x[:2] = R2 @ x[:2] + t
        x[4:6] = R2 @ x[4:6]
        P[:2, :2] = R2 @ P[:2, :2] @ R2.T
        P[4:6, 4:6] = R2 @ P[4:6, 4:6] @ R2.T
        return x, P
