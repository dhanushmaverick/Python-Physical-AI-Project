def run_motion(ctx):
    """Stack order requested by user: Blue on green and red on blue.

    Final stack:
    bottom = green
    middle = blue
    top = red

    Important:
    - Add +90 to the true yaw angles of the blocks in code for gripper alignment.
    - Block object positions are already available/consistent with the current RoboDK context.
    """

    robot = ctx.robot

    # ------------------------------------------------------------
    # World coordinates of blocks (from current query)
    # ------------------------------------------------------------
    red_x, red_y = 20.162, 9.737
    red_yaw = 91.0783 + 90

    green_x, green_y = 161.319, 75.452
    green_yaw = 14.3145 + 90

    blue_x, blue_y = 81.481, 49.459
    blue_yaw = 52.3926 + 90

    # Z parameters
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # Use green's footprint as the stack location
    stack_x = green_x
    stack_y = green_y
    stack_orient = green_yaw

    green_place_z = place_base_z
    blue_place_z = place_base_z + block_thickness
    red_place_z = place_base_z + block_thickness * 2.0

    # ------------------------------------------------------------
    # Pick GREEN (bottom)
    # ------------------------------------------------------------
    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, pick_z, green_yaw))

    ctx.close_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))

    # Place GREEN
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, green_place_z, stack_orient))

    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick BLUE (middle)
    # ------------------------------------------------------------
    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))

    ctx.close_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # Place BLUE on GREEN
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, stack_orient))

    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick RED (top)
    # ------------------------------------------------------------
    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))

    ctx.close_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # Place RED on BLUE
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, stack_orient))

    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # Return home
    ctx.go_home()
