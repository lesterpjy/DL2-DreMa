import time

import numpy as np
import torch
import cv2

# Kabsch algorithm
# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised.
def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

def build_quaternion(rotation_matrix):
    """Convert a 3x3 rotation matrix to a unit quaternion."""
    # Extract rotation part of the matrix
    R = rotation_matrix[..., :3, :3]

    # Compute quaternion components
    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-12)) / 2.0
    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)
    return quaternion

def prepare_depth(depth):
    """Prepare depth map for visualization."""
    mi = np.min(depth)
    ma = np.max(depth)
    depth_map_view = (depth - mi) / (ma - mi + 1e-8)
    depth_map_view = (255 * depth_map_view).astype(np.uint8)
    depth_map_view = cv2.applyColorMap(depth_map_view, cv2.COLORMAP_JET)

    return depth_map_view


class LoopFrequencyLogger:
    def __init__(self, log_interval=1.0):
        """
        Initialize the frequency logger.
        :param log_interval: Time interval (seconds) between frequency logs.
        """
        self.log_interval = log_interval
        self.start_time = time.time()
        self.iteration_count = 0

    def log_frequency(self):
        """
        Increment iteration count and log frequency when interval elapses.
        """
        self.iteration_count += 1
        current_time = time.time()

        if current_time - self.start_time >= self.log_interval:
            frequency = self.iteration_count / self.log_interval
            print(f"Loop frequency: {frequency:.2f} Hz")

            # Reset counters
            self.start_time = current_time
            self.iteration_count = 0