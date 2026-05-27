#run command :
#       python -m source_code.RoboDK_pran.RoboDK_config

import json
import math
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT
from robodk.robomath import *

from source_code.utility.paths import ROBO_DK_STATION_PATH


class RoboDKContext:
    def __init__(self):

        print("[INFO] Connecting to RoboDK...")
        self.rdk = Robolink() # Connects to the RoboDK API and starts the RoboDK application if it is not already running.
    # RoboDK should be present in the local machine under the default installation path. If RoboDK is not found, an error will be raised.
    #if the default installation path is not used, the path to RoboDK should be added to the system PATH variable and the RoboDK API should be installed via pip.
    # USE WITH CAUTION : rdk = Robolink(robodk_path="C:/bin/RoboDK/bin/RoboDK.exe") 


        print("[INFO] Opening RoboDK station:")
        print(ROBO_DK_STATION_PATH)
        self.rdk.AddFile(str(ROBO_DK_STATION_PATH)) #this function opens the specified RoboDK station file. If the file is not found, an error will be raised.

        self.rdk.setSimulationSpeed(0.5)
        # -----------------------------
        # Main RoboDK items
        # -----------------------------
        self.robot = self.rdk.Item("UR5e", ITEM_TYPE_ROBOT) #each item in the RoboDK station has a type and a name. This function searches for an item with the specified name and type. 
    # If the item is not found, an error will be raised.

        if not self.robot.Valid():
            raise RuntimeError("Robot 'UR5e' not found in RoboDK station.")

        self.home = self.rdk.Item("HOME_main")
        self.world_frame = self.rdk.Item("WORLD_FRAME")
        self.robot.setPoseFrame(self.world_frame)
        if not self.world_frame.Valid():
            raise RuntimeError("WORLD_FRAME not found in RoboDK station.")

        # -----------------------------
        # Block objects
        # These names match your current station tree.
        # -----------------------------
        self.blocks = {
            "red": self.rdk.Item("red_object"),
            "green": self.rdk.Item("green_object"),
            "blue": self.rdk.Item("blue_object"),
        }

        self._validate_items()

        # -----------------------------
        # Motion constants
        # AI can use these values
        # -----------------------------
        self.approach_z = 150.0
        self.pick_z = 10.0
        self.place_base_z = 10.0
        self.block_thickness = 20.0

        print("[SUCCESS] RoboDK context ready.")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _validate_items(self):
        for color, obj in self.blocks.items():
            if not obj.Valid():
                raise RuntimeError(f"Missing RoboDK object: {color}_object")

        if not self.home.Valid():
            print("[WARN] HOME target not found. ctx.go_home() will be skipped.")

    # ------------------------------------------------------------
    # Pose helper
    # ------------------------------------------------------------
    def pose_xyz_yaw(self, x, y, z, yaw_deg):
        """
        Top-down TCP pose relative to WORLD_FRAME.
        """
        return (
            transl(float(x), float(y), float(z))
            * rotz(math.radians(float(yaw_deg)))
            * rotx(math.radians(180))
        )

    # ------------------------------------------------------------
    # Object placement handled by Python
    # ------------------------------------------------------------
    def set_block_pose(self, color, x, y, z= 0.0, yaw_deg=0.0):
        obj = self.blocks.get(color)

        if obj is None or not obj.Valid():
            raise RuntimeError(f"Missing RoboDK object for color: {color}")

        obj.setParentStatic(self.world_frame)
        obj.setPose(self.pose_xyz_yaw(x, y, z, yaw_deg))

        print(f"[INFO] Set {color}_object -> x={x}, y={y}, z={z}, yaw={yaw_deg}")

    def place_blocks_from_poses(self, poses):
        for color, pose in poses.items():
            self.set_block_pose(
                color,
                pose["x"],
                pose["y"],
                pose.get("z", self.block_thickness),
                pose["yaw_deg"],
            )

    def load_block_poses_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        red_pos = data["red_block_position"]
        green_pos = data["green_block_position"]
        blue_pos = data["blue_block_position"]

        return {
            "red": {
                "x": float(red_pos[0]),
                "y": float(red_pos[1]),
                "z": self.block_thickness,
                "yaw_deg": float(data["red_block_orientation"]),
            },
            "green": {
                "x": float(green_pos[0]),
                "y": float(green_pos[1]),
                "z": self.block_thickness,
                "yaw_deg": float(data["green_block_orientation"]),
            },
            "blue": {
                "x": float(blue_pos[0]),
                "y": float(blue_pos[1]),
                "z": self.block_thickness,
                "yaw_deg": float(data["blue_block_orientation"]),
            },
        }

    def place_blocks_from_json(self, json_path):
        poses = self.load_block_poses_from_json(json_path)
        self.place_blocks_from_poses(poses)
        print("[SUCCESS] Updated RoboDK block poses from JSON.")

    def place_test_blocks(self):
        poses = {
            "red": {
                "x": 150.0,
                "y": 50.0,
                "z": self.block_thickness,
                "yaw_deg": -90.0,
            },
            "green": {
                "x": 400.0,
                "y": 300.0,
                "z": self.block_thickness,
                "yaw_deg": 30.0,
            },
            "blue": {
                "x": 200.0,
                "y": 250.0,
                "z": self.block_thickness,
                "yaw_deg": 45.0,
            },
        }

        self.place_blocks_from_poses(poses)

    # ------------------------------------------------------------
    # Gripper programs
    # ------------------------------------------------------------
    def open_gripper(self,color):
        program = self.rdk.Item("HANDE_OPEN_SIM")
        obj = self.blocks.get(color)
        obj.setParentStatic(self.world_frame)
        if program.Valid():
            program.RunProgram()
            program.WaitFinished()
            
        else:
            print("[WARN] HANDE_OPEN_SIM program not found.")

    def close_gripper(self,color):
        program = self.rdk.Item("HANDE_OPEN_SIM_20mm")
        tool = self.rdk.Item("gripper_tcp")
        obj = self.blocks.get(color)
        obj.setParentStatic(tool)
        if program.Valid():
            program.RunProgram()
            program.WaitFinished()  # Wait until the program finishes before proceeding
            program.WaitFinished()
        else:
            print("[WARN] HandE_OPEN_SIM_20mm program not found.")
            print("[WARN] HANDE_CLOSE_SIM program not found.")
    

    # ------------------------------------------------------------
    # Home
    # ------------------------------------------------------------
    def go_home(self):
        if self.home.Valid():
            print("[INFO] Moving robot to HOME...")
            self.robot.MoveJ(self.home)
            #self.rdk.Finish()
            print("[INFO] Moved robot to HOME...")
        else:
            print("[WARN] HOME target not found. Skipping go_home().")