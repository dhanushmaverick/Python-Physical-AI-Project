"""
Run:
    python -m source_code.vision.calibration.run_calibration
Usage: 
    Saves calibration matrix inside data
"""
from source_code.vision.calibration.calibration import CameraCalibrator
from source_code.utility.paths import (
    RAW_IMAGES_DIR,
    INTRINSICS_FILE,
    REPORT_FILE,
    ensure_directories,
)


def main():
    ensure_directories()

    calibrator = CameraCalibrator(
        checkerboard_size=(8, 6),   # inner corners
        square_size_mm=30.0,  
    )

    report = calibrator.calibrate_from_images(
        images_dir=RAW_IMAGES_DIR,
        output_file=INTRINSICS_FILE,
        report_file=REPORT_FILE,
    )

    print("\nCalibration successful.")
    print(f"Valid images: {report['num_valid_images']}/{report['num_total_images']}")
    print(f"RMS error: {report['rms_error']:.6f}")
    print(f"Mean reprojection error: {report['mean_reprojection_error']:.6f}")
    print(f"Saved intrinsics to: {INTRINSICS_FILE}")
    print(f"Saved report to: {REPORT_FILE}")


if __name__ == "__main__":
    main()