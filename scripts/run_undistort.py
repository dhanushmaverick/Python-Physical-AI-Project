from app.utils.paths import (
    RAW_IMAGES_DIR,
    INTRINSICS_FILE,
    REPORT_FILE,
    ensure_directories,
)
from app.vision.undistort import load_intrinsics, visualize_undistortion
import cv2


intrinsics_file = INTRINSICS_FILE
image_file = RAW_IMAGES_DIR / "Im_L_1.png"  # change this to test different images

camera_matrix, dist_coeffs = load_intrinsics(intrinsics_file)
frame = cv2.imread(str(image_file))

visualize_undistortion(frame, camera_matrix, dist_coeffs)