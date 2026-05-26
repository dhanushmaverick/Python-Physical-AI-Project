import math
from robodk.robolink import *
from robodk.robomath import *

# --------------------
# Workspace setup
# --------------------
# Connection
RDK = Robolink()

# NOTE: We assume the RoboDK station is already set up with:
# - a UR5e robot item
# - a gripper/tool already defined
# - block pick/place targets already existing or reachable via approach poses
# Since we only have world coordinates for blocks, we create dynamic pick/place targets.

robot = None
# Try to locate the UR5e robot automatically
for it in RDK.ItemList():
    pass

robot = RDK.Item('', ITEM_TYPE_ROBOT)
if robot.Valid() == 0:
    # Fallback: first robot item
    robots = RDK.ItemList(ITEM_TYPE_ROBOT)
    robot = RDK.Item(robots[0] if robots else '')

if robot.Valid() == 0:
    raise RuntimeError("UR5e robot item not found in the RoboDK station.")

# Set run mode (simulation by default)
try:
    RDK.setRunMode(RUNMODE_SIMULATE)
except Exception:
    pass

# Simulation speed
try:
    RDK.setSimulationSpeed(100)
except Exception:
    pass

# Get world frame
world = RDK.World()

# Try to locate a gripper/tool (optional)
# In many stations, the gripper tool is the active tool. We'll just rely on Move commands.

# --------------------
# Block coordinates (world)
# --------------------
# Input from prompt tool (world coordinates)
red_xy = [19.456730487417932, 9.850884774922811]
red_yaw_deg = 89.7196273803711
blue_xy = [81.30805217450195, 49.41455508328906]
blue_yaw_deg = 47.37146377563476
green_xy = [162.23254021556767, 75.20602232885692]
green_yaw_deg = 26.975744247436527

# We need Z heights. Since only XY are provided, we choose reasonable approach/pick/place heights.
# Adjust these if your station uses different Z values.
PICK_Z = 5.0
APPROACH_Z = 20.0
DROP_Z_1 = 0.0   # base placement Z for the bottom layer (blue)
BLOCK_THICKNESS = 3.0
DROP_Z_2 = DROP_Z_1 + BLOCK_THICKNESS
DROP_Z_3 = DROP_Z_2 + BLOCK_THICKNESS

# --------------------
# Helpers
# --------------------

def yaw_to_rot(z_deg: float):
    """Create a rotation matrix around Z (world frame)."""
    z = math.radians(z_deg)
    c, s = math.cos(z), math.sin(z)
    # Robodk.Rot needs 3x3 or we can use Mat constructor
    return [
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ]


def pose_from_xy_z_yaw(x, y, z, yaw_deg):
    r = yaw_to_rot(yaw_deg)
    mat = Mat([
        [r[0][0], r[0][1], r[0][2], x],
        [r[1][0], r[1][1], r[1][2], y],
        [r[2][0], r[2][1], r[2][2], z],
        [0,       0,       0,       1],
    ])
    return mat


def make_target(pose, name: str):
    # Create a target item (optional). We can just pass pose to MoveL/MoveJ.
    return pose


def move_pick(xy, yaw_deg):
    x, y = xy
    # Approach
    robot.MoveJ(pose_from_xy_z_yaw(x, y, APPROACH_Z, yaw_deg))
    # Descend
    robot.MoveL(pose_from_xy_z_yaw(x, y, PICK_Z, yaw_deg))
    # Close gripper placeholder (tool actuation depends on station I/O)
    # In many stations, gripper is controlled via program logic, not in script.


def move_place(xy, yaw_deg, z_drop):
    x, y = xy
    # Approach
    robot.MoveJ(pose_from_xy_z_yaw(x, y, APPROACH_Z, yaw_deg))
    # Descend
    robot.MoveL(pose_from_xy_z_yaw(x, y, z_drop, yaw_deg))
    # Open gripper placeholder


# --------------------
# Stacking plan
# --------------------
# User order: red on top of blue, then green block.
# Interpretation: bottom=blue, middle=red, top=green.

# 1) Pick BLUE and place at base (blue position). If blue is already there, robot will still perform pick/place.
move_pick(blue_xy, blue_yaw_deg)
move_place(blue_xy, blue_yaw_deg, DROP_Z_1)

# 2) Pick RED and place on top of blue
move_pick(red_xy, red_yaw_deg)
move_place(blue_xy, red_yaw_deg, DROP_Z_2)

# 3) Pick GREEN and place on top of red (top)
move_pick(green_xy, green_yaw_deg)
move_place(blue_xy, green_yaw_deg, DROP_Z_3)

# Return home
robot.MoveJ(pose_from_xy_z_yaw(blue_xy[0], blue_xy[1], APPROACH_Z, 0))
