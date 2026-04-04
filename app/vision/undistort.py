from pathlib import Path
import cv2
import numpy as np


def load_intrinsics(intrinsics_file: Path):
    if not intrinsics_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {intrinsics_file}")

    data = np.load(intrinsics_file, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def undistort_frame(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs)