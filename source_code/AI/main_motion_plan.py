def run_motion(ctx):
    """Stacking order: blue on red, then green on blue.

    All blocks are placed with the same orientation.
    Note: +90 is added to the true yaw angles for lateral gripper alignment.
    """

    robot = ctx.robot

    # ------------------------------------------------------------
    # Current block poses (from World coords / station setup)
    # ------------------------------------------------------------
    red_x, red_y = 20.162, 9.737
    red_yaw = 91.0783 + 90

    blue_x, blue_y = 81.481, 49.459
    blue_yaw = 52.3926 + 90

    green_x, green_y = 161.319, 75.452
    green_yaw = 14.3145 + 90

    # ------------------------------------------------------------
    # Z values from RoboDKContext
    # ------------------------------------------------------------
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # ------------------------------------------------------------
    # Use RED as the base location for the final stack
    # ------------------------------------------------------------
    stack_x = red_x
    stack_y = red_y

    # Use consistent stack orientation (same for all stacked blocks)
    # Choose red yaw as the reference.
    stack_orient = red_yaw

    red_place_z = place_base_z
    blue_place_z = place_base_z + block_thickness
    green_place_z = place_base_z + block_thickness * 2.0

    # ------------------------------------------------------------
    # Pick RED (place onto stack base)
    # ------------------------------------------------------------
    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))
    ctx.close_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # Place RED at base
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, stack_orient))
    ctx.open_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick BLUE and place on RED
    # ------------------------------------------------------------
    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))
    ctx.close_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # Place BLUE on RED
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, stack_orient))
    ctx.open_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick GREEN and place on BLUE
    # ------------------------------------------------------------
    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, pick_z, green_yaw))
    ctx.close_gripper("green")
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))

    # Place GREEN on BLUE
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, green_place_z, stack_orient))
    ctx.open_gripper("green")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Return home
    # ------------------------------------------------------------
    ctx.go_home()
