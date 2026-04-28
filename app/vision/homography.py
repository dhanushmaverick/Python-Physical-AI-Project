import numpy as np
import cv2


def compute_homography(world_points: np.ndarray, image_points: np.ndarray):
    H, status = cv2.findHomography(world_points, image_points)
    return H


def image_to_world(image_point: np.ndarray, H: np.ndarray):
    H_inv = np.linalg.inv(H)

    pt = np.array([image_point[0], image_point[1], 1.0])
    world_pt = H_inv @ pt
    world_pt /= world_pt[2]

    return world_pt[:2]


def world_to_image(world_point: np.ndarray, H: np.ndarray):
    pt = np.array([world_point[0], world_point[1], 1.0])
    image_pt = H @ pt
    image_pt /= image_pt[2]

    return image_pt[:2]