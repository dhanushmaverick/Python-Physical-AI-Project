#run command:
#       python -m source_code.vision.homography.run_image_to_world_transformation


from source_code.utility.paths import *
from source_code.vision.homography.homography import image_to_world
import json
import numpy as np


with open(IMG_COORDINATE_DATA / "Pose.json", "r") as file:
    pose_data = json.load(file)

with open(HOMOGRAPHY_REPORT, "r") as file:
    Homography_data = json.load(file)
    H = np.array(Homography_data["Homography_Matrix"])




world_data = {}

for color in ["red", "green", "blue"]:
    image_position = np.array(pose_data[f"{color}_block_position"], dtype=float)
    orientation = pose_data[f"{color}_block_orientation"]

    world_position = image_to_world(image_position, H)

    world_data[f"{color}_block_position"] = world_position.tolist()
    world_data[f"{color}_block_orientation"] = round(orientation,4)


with open(HOMOGRAPHY_DATA / "World_Pose.json", "w") as file:
    json.dump(world_data, file, indent=4)

print(world_data)