from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CODE_DIR = PROJECT_ROOT / "source_code"
VISION_DIR = SOURCE_CODE_DIR / "vision"
CALIBRATION_DIR = VISION_DIR / "calibration"
HOMOGRAPHY_DIR = VISION_DIR / "homography"
HOMOGRAPHY_DATA = HOMOGRAPHY_DIR / "data"
DATA_DIR = CALIBRATION_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
OBJ_SEGMENTATION_DIR = VISION_DIR / "object_segmentation"
IMG_COORDINATE_DATA = OBJ_SEGMENTATION_DIR / "data"

AI_DIR = SOURCE_CODE_DIR / "AI"
AI_MOTION_PLAN_PATH = AI_DIR / "main_motion_plan.py"

ROBODK_DIR = SOURCE_CODE_DIR / "simulation"
ROBO_DK_STATION_PATH = SOURCE_CODE_DIR / "RoboDKSIM.rdk"


INTRINSICS_FILE = DATA_DIR/ "camera_intrinsics.npz"
REPORT_FILE = DATA_DIR/ "calibration_report.json"
HOMOGRAPHY_REPORT = HOMOGRAPHY_DATA /  "homography_report.json"
WORLD_POSE = HOMOGRAPHY_DATA / "World_Pose.json"


def ensure_directories() -> None:
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    