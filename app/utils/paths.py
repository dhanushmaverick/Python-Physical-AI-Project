from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CALIBRATION_DIR = DATA_DIR / "calibration"
RAW_IMAGES_DIR = CALIBRATION_DIR / "raw_images"

INTRINSICS_FILE = CALIBRATION_DIR / "camera_intrinsics.npz"
REPORT_FILE = CALIBRATION_DIR / "calibration_report.json"


def ensure_directories() -> None:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)