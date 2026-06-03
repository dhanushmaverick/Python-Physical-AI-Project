def run_motion(ctx):
    """Stack red on green, then place blue."""

    robot = ctx.robot

    # Current world coordinates (from tool get_world_coords_string)
    red_x, red_y = 73.965, 30.888
    red_yaw = 30.2839 + 90  # +90 per requirement

    green_x, green_y = 50.318, 76.553
    green_yaw = 96.4585 + 90  # +90 per requirement

    blue_x, blue_y = 8.789, 54.985
    blue_yaw = 149.3678 + 90  # +90 per requirement

    # Z parameters from context
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # Use GREEN as the stacking base
    stack_x = green_x
    stack_y = green_y
    stack_yaw = green_yaw

    green_place_z = place_base_z
    red_place_z = place_base_z + block_thickness
    blue_place_z = place_base_z + block_thickness * 2.0

    # ------------------------------------------------------------
    # Pick RED (to place on GREEN)
    # ------------------------------------------------------------
    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))
    ctx.close_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # ------------------------------------------------------------
    # Place RED on GREEN
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, stack_yaw))
    ctx.open_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_yaw))

    # ------------------------------------------------------------
    # Pick BLUE (to place on top)
    # ------------------------------------------------------------
    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))
    ctx.close_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # ------------------------------------------------------------
    # Place BLUE on RED
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, stack_yaw))
    ctx.open_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_yaw))

    ctx.go_home()
