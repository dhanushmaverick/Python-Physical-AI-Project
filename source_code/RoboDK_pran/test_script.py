def run_motion(ctx):
    """
    Test AI-generated motion plan.

    Final stack:
    bottom = green
    middle = red
    top = blue

    Important:
    - Block object positions are already updated from World_Pose.json
      by the RoboDK Python side.
    - This script only performs robot motion.
    """

    robot = ctx.robot

    # ------------------------------------------------------------
    # Block poses from current World_Pose.json
    # ------------------------------------------------------------
    red_x = 19.924
    red_y = 9.888
    red_yaw = 89.7196

    green_x = 161.474
    green_y = 75.212
    green_yaw = 26.9757

    blue_x = 81.298
    blue_y = 49.511
    blue_yaw = 47.3715

    # ------------------------------------------------------------
    # Use Z values from RoboDKContext
    # These prevent the robot from going under the table.
    # ------------------------------------------------------------
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # Stack location: use green block as the base location
    stack_x = green_x
    stack_y = green_y

    green_place_z = place_base_z
    red_place_z = place_base_z + block_thickness
    blue_place_z = place_base_z + block_thickness * 2.0

    # ------------------------------------------------------------
    # Pick GREEN
    # ------------------------------------------------------------
    ctx.open_gripper("green")

    robot.MoveJ(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, pick_z, green_yaw))

    ctx.close_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))

    # ------------------------------------------------------------
    # Place GREEN as bottom block
    # ------------------------------------------------------------
    robot.MoveJ(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, green_place_z, green_yaw))

    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, green_yaw))

    # ------------------------------------------------------------
    # Pick RED
    # ------------------------------------------------------------
    ctx.open_gripper("red")

    robot.MoveJ(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))

    ctx.close_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # ------------------------------------------------------------
    # Place RED on GREEN
    # ------------------------------------------------------------
    robot.MoveJ(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, red_yaw))

    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, red_yaw))

    # ------------------------------------------------------------
    # Pick BLUE
    # ------------------------------------------------------------
    ctx.open_gripper("blue")

    robot.MoveJ(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))

    ctx.close_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # ------------------------------------------------------------
    # Place BLUE on RED
    # ------------------------------------------------------------
    robot.MoveJ(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, blue_yaw))

    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, blue_yaw))

    # ------------------------------------------------------------
    # Return home
    # ------------------------------------------------------------
    ctx.go_home()