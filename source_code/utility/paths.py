from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CODE_DIR = PROJECT_ROOT / "source_code"
VISION_DIR = SOURCE_CODE_DIR / "vision"
CALIBRATION_DIR = VISION_DIR / "calibration"
DATA_DIR = CALIBRATION_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
OBJ_SEGMENTATION_DIR = VISION_DIR / "object_segmentation"


INTRINSICS_FILE = DATA_DIR/ "camera_intrinsics.npz"
REPORT_FILE = DATA_DIR/ "calibration_report.json"
HOMOGRAPHY_FILE = DATA_DIR / "homography.npz"
HOMOGRAPHY_REPORT = DATA_DIR / "homography_report.json"


def ensure_directories() -> None:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    HOMOGRAPHY_FILE.parent.mkdir(parents=True, exist_ok=True)
    