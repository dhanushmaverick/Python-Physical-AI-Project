import json
from pathlib import Path
import cv2
import numpy as np


class CameraCalibrator:
    def __init__(
        self,
        checkerboard_size: tuple[int, int] = (11, 7),  #change depending on the checkerboard used (number of inner corners) 1 less than the number of squares in each dimension
        square_size_mm: float = 30.0,   #size in mm
    ):
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm

    def _create_object_points(self) -> np.ndarray:  ## excpected to return a (N, 3) array of 3D points in the checkerboard's coordinate system
        cols, rows = self.checkerboard_size
        objp = np.zeros((rows * cols, 3), np.float32)   ## using np.float32 to ensure compatibility with OpenCV functions
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  #grid of points in the checkerboard plane
        objp *= self.square_size_mm # scale the points to the actual size of the squares in millimeters
        return objp       # basically returns a ideal grid of 3D points corresponding to the inner corners of the checkerboard, which will be used as the reference for calibration

    def calibrate_from_images(self, images_dir: Path, output_file: Path, report_file: Path) -> dict:
        image_paths =  sorted(images_dir.glob("*.png"))  # looking for all PNG images in the specified directory; can be changed to other formats if needed (e.g., "*.jpg")
        if not image_paths:
            raise FileNotFoundError(f"No calibration images found in: {images_dir}")

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )  # standard criteria for corner refinement: either 30 iterations or an accuracy of 0.001, whichever comes first

        objp = self._create_object_points()

        objpoints = [] #object points in 3D
        imgpoints = []  # corresponding image points in 2D
        valid_images = [] # list of images where corners were successfully detected and used for calibration
        invalid_images = [] # list of images where corners could not be detected, which will be reported back to the user for troubleshooting 
        image_size = None

        for image_path in image_paths: # iterating through each image found 
            image = cv2.imread(str(image_path))
            if image is None:
                invalid_images.append(str(image_path))
                continue    #skipped if the image cannot be read

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #image to grayscale for corner detection, as the algorithm works better on single channel images
            image_size = gray.shape[::-1] # storing the image size (width, height) for later use in calibration; shape gives (height, width), so we reverse it to get (width, height)
                                          #openCV expects the image size in (width, height) format, which is why we reverse

            # preprocessing 
            # gray = cv2.equalizeHist(gray)   # improves contrast
            # #gray = cv2.GaussianBlur(gray, (5, 5), 0)   # reduces noise slightly
            # # kernel = np.array([[0, -1, 0],
            # #                    [-1, 5, -1],
            # #                    [0, -1, 0]])
            # # gray = cv2.filter2D(gray, -1, kernel)   # sharpens edges a bit
            # cv2.imshow("processed", gray)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            ## problematic line:
            found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)  #returns a boolean if found and corner locations
            # if found:
            # print(found)

            # for i in corners:
            #     cv2.circle(image, (i[0][0], i[0][1]), 5, (0, 255, 0), -1)  #visualize the detected corners on the image for debugging purposes; draws a green circle at each detected corner

            if found:
                refined_corners = cv2.cornerSubPix(  #refining the coreners to sub-pixel accuracy, which can improve calibration results
                    gray,
                    corners,
                    (11, 11), 
                    (-1, -1),
                    criteria, # termination criteria which was defined above
                )
                print(f"Found corners in image: {image_path.name}")  #debugging output to indicate which images were successfully processed

                objpoints.append(objp)
                imgpoints.append(refined_corners)
                valid_images.append(str(image_path))
            else:
                invalid_images.append(str(image_path))
                print(f"Warning: Could not find corners in image: {image_path.name}")  #debugging output to indicate which images were skipped due to corner detection failure

        if len(valid_images) < 10:
            raise RuntimeError(
                f"Not enough valid calibration images. Found {len(valid_images)} valid images; need at least 10."
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(  #standard opencv function returns the RMS re-projection error, camera matrix, distortion coefficients, and the rotation and translation vectors for each image
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )

        mean_reprojection_error = self._compute_mean_reprojection_error(  
            objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
        )   #custom quality check

        np.savez( #saving the calibration results in a .npz file format in a numpy archive
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