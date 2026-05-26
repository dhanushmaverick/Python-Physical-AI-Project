import math
from source_code.utility.paths import *

from robodk.robolink import *
from robodk.robomath import *


# Change this to your actual RoboDK station path



class RoboDKRunner:
    def __init__(self):
        self.RDK = Robolink()

        # Open specific RoboDK workspace/station
        

        if not ROBO_DK_STATION_PATH.exists():
            raise FileNotFoundError(f"RoboDK station file not found: {ROBO_DK_STATION_PATH}")

        self.RDK.AddFile(str(ROBO_DK_STATION_PATH))

        self.robot = self.RDK.Item("UR5e", ITEM_TYPE_ROBOT)

        if not self.robot.Valid():
            raise RuntimeError("Robot 'UR5e' not found in RoboDK station.")

        self.home = self.RDK.Item("HOME")

        self.blocks = {
            "red": {
                "frame": self.RDK.Item("WORLD_FRAME"),
                "object": self.RDK.Item("red_object"),
            },
            "green": {
                "frame": self.RDK.Item("WORLD_FRAME"),
                "object": self.RDK.Item("green_object"),
            },
            "blue": {
                "frame": self.RDK.Item("WORLD_FRAME"),
                "object": self.RDK.Item("blue_object"),
            },
        }

        self.approach_z = 50
        self.pick_z = 5

    def pose(self, x, y, z, yaw_deg=0):
        return transl(x, y, z) * rotz(math.radians(yaw_deg))

    def update_block_pose(self, color, x, y, z=0, yaw_deg=0):
        block_frame = self.blocks[color]["frame"]

        if not block_frame.Valid():
            raise RuntimeError(f"Block frame for '{color}' not found.")

        block_frame.setPose(self.pose(x, y, z, yaw_deg))

    def move_to(self, x, y, z, yaw_deg=0):
        self.robot.MoveJ(self.pose(x, y, z, yaw_deg))

    def move_linear(self, x, y, z, yaw_deg=0):
        self.robot.MoveL(self.pose(x, y, z, yaw_deg))

    def open_gripper(self):
        prog = self.RDK.Item("OpenGripper")
        if prog.Valid():
            prog.RunProgram()
        else:
            print("[WARN] OpenGripper program not found.")

    def close_gripper(self):
        prog = self.RDK.Item("CloseGripper")
        if prog.Valid():
            prog.RunProgram()
        else:
            print("[WARN] CloseGripper program not found.")

    def pick(self, color, x, y, yaw_deg=0):
        self.open_gripper()
        self.move_to(x, y, self.approach_z, yaw_deg)
        self.move_linear(x, y, self.pick_z, yaw_deg)
        self.close_gripper()
        self.move_linear(x, y, self.approach_z, yaw_deg)

    def place(self, x, y, z=0, yaw_deg=0):
        self.move_to(x, y, self.approach_z, yaw_deg)
        self.move_linear(x, y, z, yaw_deg)
        self.open_gripper()
        self.move_linear(x, y, self.approach_z, yaw_deg)

    def go_home(self):
        if self.home.Valid():
            self.robot.MoveJ(self.home)
        else:
            print("[WARN] HOME target not found.")