from pathlib import Path
import cv2
import numpy as np


print("Loading camera intrinsics...")

def load_intrinsics(intrinsics_file: Path):
    if not intrinsics_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {intrinsics_file}")

    data = np.load(intrinsics_file, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def undistort_frame(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def visualize_undistortion(frame, camera_matrix, dist_coeffs):
    undistorted = undistort_frame(frame, camera_matrix, dist_coeffs)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Undistorted Frame", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

