"""
Usage:
    python -m source_code.vision.calibration.run_capture_images

Description:
    This script opens a live video stream from the connected camera and allows
    the user to capture calibration images manually.

Purpose:
    The captured images are typically used for camera calibration. For better
    calibration accuracy, capture images from different positions, angles,
    and distances.

Controls:
    s  -> Save/Capture the current frame as an image
    q  -> Quit the video stream and exit the program

Recommended Procedure:
    1. Run the script using the command above.
    2. Wait for the camera stream window to open.
    3. Hold the calibration pattern (e.g., checkerboard) in front of the camera.
    4. Move and rotate the pattern to different orientations and positions.
    5. Press 's' to capture each image.
    6. Capture approximately 15 - 20 images for reliable calibration.
    7. Press 'q' once finished to close the application.

Example:
    python -m scripts.capture_images
"""
# Run camera calibration after this

from source_code.utility.paths import RAW_IMAGES_DIR, ensure_directories
import cv2

ensure_directories()

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = RAW_IMAGES_DIR / f"img_{count:02d}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()