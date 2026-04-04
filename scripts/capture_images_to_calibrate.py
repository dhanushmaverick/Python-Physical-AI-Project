import json
from pathlib import Path
import cv2
import numpy as np


class CameraCalibrator:
    def __init__(
        self,
        checkerboard_size: tuple[int, int] = (6, 3),
        square_size_mm: float = 30.0,
    ):
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm

    def _create_object_points(self) -> np.ndarray:
        cols, rows = self.checkerboard_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp

    def calibrate_from_images(self, images_dir: Path, output_file: Path, report_file: Path) -> dict:
        image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        if not image_paths:
            raise FileNotFoundError(f"No calibration images found in: {images_dir}")

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        objp = self._create_object_points()

        objpoints = []
        imgpoints = []
        valid_images = []
        invalid_images = []

        image_size = None

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                invalid_images.append(str(image_path))
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]

            found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if found:
                refined_corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria,
                )

                objpoints.append(objp)
                imgpoints.append(refined_corners)
                valid_images.append(str(image_path))
            else:
                invalid_images.append(str(image_path))

        if len(valid_images) < 10:
            raise RuntimeError(
                f"Not enough valid calibration images. Found {len(valid_images)} valid images; need at least 10."
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )

        mean_reprojection_error = self._compute_mean_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
        )

        np.savez(
            output_file,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=np.array(rvecs, dtype=object),
            tvecs=np.array(tvecs, dtype=object),
            image_width=image_size[0],
            image_height=image_size[1],
            checkerboard_cols=self.checkerboard_size[0],
            checkerboard_rows=self.checkerboard_size[1],
            square_size_mm=self.square_size_mm,
        )

        report = {
            "num_total_images": len(image_paths),
            "num_valid_images": len(valid_images),
            "num_invalid_images": len(invalid_images),
            "valid_images": valid_images,
            "invalid_images": invalid_images,
            "rms_error": float(rms),
            "mean_reprojection_error": float(mean_reprojection_error),
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "checkerboard_size": {
                "cols": self.checkerboard_size[0],
                "rows": self.checkerboard_size[1],
            },
            "square_size_mm": self.square_size_mm,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return report

    @staticmethod
    def _compute_mean_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
    ) -> float:
        total_error = 0.0

        for i in range(len(objpoints)):
            projected_points, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error

        return total_error / len(objpoints)