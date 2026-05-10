"""
Kalman Filter for 2D bounding box tracking.

State vector (8-dim): [cx, cy, a, h,  vcx, vcy, va, vh]
  cx, cy  - center of bounding box
  a       - aspect ratio (w / h)
  h       - height
  v*      - velocities

Measurement (4-dim): [cx, cy, a, h]
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    # Chi-squared quantile (p=0.95) for gating:
    #   4-dim measurements → 9.4877
    #   2-dim (position only) → 5.9915
    chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}

    def __init__(self):
        ndim, dt = 4, 1

        # State transition matrix: constant velocity model
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Measurement matrix: observe position only
        self.H = np.eye(ndim, 2 * ndim)

        # Uncertainty weights (from DeepSORT paper)
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement):
        """Create initial state from first measurement [cx, cy, a, h]."""
        mean = np.r_[measurement, np.zeros_like(measurement)]

        std = [
            2 * self._std_weight_pos * measurement[3],
            2 * self._std_weight_pos * measurement[3],
            1e-2,
            2 * self._std_weight_pos * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            1e-5,
            10 * self._std_weight_vel * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Advance state one time step (no measurement)."""
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3],
        ]
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3],
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def project(self, mean, covariance):
        """Project state into measurement space."""
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        R = np.diag(np.square(std))

        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + R
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """Correct state using new measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain: K = P H^T S^{-1}  →  K^T = S^{-1} H P  (cho_solve over (4,8))
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), self.H @ covariance, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Mahalanobis distance from predicted position to a set of measurements.
        Used to gate implausible associations before Hungarian matching.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        L = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean                        # (N, dim)
        z = scipy.linalg.solve_triangular(L, d.T, lower=True, check_finite=False)
        return np.sum(z * z, axis=0)                            # (N,)
