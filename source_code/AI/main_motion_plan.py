import math
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT
from robodk.robomath import *

# ------------------------------------------------------------
# Auto-generated motion plan: Stack red on green, then blue.
# ------------------------------------------------------------

RDK = Robolink()

# Locate a UR5 robot in the station (name may vary)
robot = None
for name in ['UR5', 'UR5e', 'UR5_1', 'UR5e (UR5)', 'UR5e Robot']:
    try:
        itm = RDK.Item(name, ITEM_TYPE_ROBOT)
        if itm.Valid():
            robot = itm
            break
    except:
        pass
if robot is None:
    robot = RDK.Item('', ITEM_TYPE_ROBOT)

# World (XY) positions provided
red_xy = [19.456730487417932, 9.850884774922811]
red_yaw_deg = 89.7196273803711

green_xy = [162.23254021556767, 75.20602232885692]
green_yaw_deg = 26.975744247436527 

blue_xy = [81.30805217450195, 49.41455508328906]
blue_yaw_deg = 47.37146377563476

# Workspace/stack assumptions (adjust to your setup if needed)
TABLE_Z = 0.0
APPROACH_Z = 80.0     # safe height above pick/place (mm)
PICK_Z = 0.0          # pick Z at block top/center plane (mm)

BLOCK_THICKNESS = 20.0  # mm (adjust to match your scene)
def pose_from_xy_yaw(x, y, z, yaw_deg):
    """Pose with rotation about Z by yaw_deg and translation (x,y,z)."""
    yaw = math.radians(yaw_deg)
    Rz = [[math.cos(yaw), -math.sin(yaw), 0],
          [math.sin(yaw),  math.cos(yaw), 0],
          [0,              0,             1]]
    return transl(x, y, z) * Mat(Rz[0][0], Rz[0][1], Rz[0][2], 0,
                               Rz[1][0], Rz[1][1], Rz[1][2], 0,
                               Rz[2][0], Rz[2][1], Rz[2][2], 0,
                               0,         0,         0,         1)

# Pick poses (above + at)
red_pick_app = pose_from_xy_yaw(red_xy[0], red_xy[1], APPROACH_Z, red_yaw_deg)
red_pick     = pose_from_xy_yaw(red_xy[0], red_xy[1], PICK_Z, red_yaw_deg)

green_pick_app = pose_from_xy_yaw(green_xy[0], green_xy[1], APPROACH_Z, green_yaw_deg)
green_pick     = pose_from_xy_yaw(green_xy[0], green_xy[1], PICK_Z, green_yaw_deg)

blue_pick_app = pose_from_xy_yaw(blue_xy[0], blue_xy[1], APPROACH_Z, blue_yaw_deg)
blue_pick     = pose_from_xy_yaw(blue_xy[0], blue_xy[1], PICK_Z, blue_yaw_deg)

# Stacking target (base aligned to green XY)
base_x, base_y = green_xy[0], green_xy[1]
z_green_place = TABLE_Z + BLOCK_THICKNESS / 2.0
z_red_place   = z_green_place + BLOCK_THICKNESS
z_blue_place  = z_red_place + BLOCK_THICKNESS

green_place_app = pose_from_xy_yaw(base_x, base_y, APPROACH_Z, green_yaw_deg)
green_place     = pose_from_xy_yaw(base_x, base_y, z_green_place, green_yaw_deg)

red_place_app = pose_from_xy_yaw(base_x, base_y, APPROACH_Z, red_yaw_deg)
red_place     = pose_from_xy_yaw(base_x, base_y, z_red_place, red_yaw_deg)

blue_place_app = pose_from_xy_yaw(base_x, base_y, APPROACH_Z, blue_yaw_deg)
blue_place     = pose_from_xy_yaw(base_x, base_y, z_blue_place, blue_yaw_deg)

# Build program
program = robot.ProgStart('Stack_R_on_G_then_B')

# ---- Pick & place GREEN (bottom) ----
robot.MoveL(green_pick_app, False)
robot.MoveL(green_pick, False)
robot.MoveL(green_pick_app, False)               # TODO: close gripper here
robot.MoveL(green_place_app, False)
robot.MoveL(green_place, False)
robot.MoveL(green_place_app, False)             # TODO: open gripper here

# ---- Pick & place RED (middle) ----
robot.MoveL(red_pick_app, False)
robot.MoveL(red_pick, False)
robot.MoveL(red_pick_app, False)                 # TODO: close gripper here
robot.MoveL(red_place_app, False)
robot.MoveL(red_place, False)
robot.MoveL(red_place_app, False)               # TODO: open gripper here

# ---- Pick & place BLUE (top) ----
robot.MoveL(blue_pick_app, False)
robot.MoveL(blue_pick, False)
robot.MoveL(blue_pick_app, False)                # TODO: close gripper here
robot.MoveL(blue_place_app, False)
robot.MoveL(blue_place, False)
robot.MoveL(blue_place_app, False)              # TODO: open gripper here

robot.ProgFinish(program)

# Uncomment to run immediately (if desired)
# RDK.RunProgram(program, True)