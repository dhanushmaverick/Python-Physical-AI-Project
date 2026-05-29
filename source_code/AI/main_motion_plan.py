def run_motion(ctx):
    """Stack order per user:
    1) Red placed on top of Blue (blue -> base, red -> middle)
    2) Green placed on top of Red (green -> top)

    All blocks are stacked with the same orientation.
    Note: Add +90 to the true yaw angles of the blocks in order for gripper to align laterally for gripping.
    """

    robot = ctx.robot

    # ------------------------------------------------------------
    # Read block poses (world-frame) from the RoboDKContext example values
    # (Yaw angles: +90 applied as required)
    # ------------------------------------------------------------
    red_x, red_y = 19.924, 9.888
    red_yaw = 91.0783 + 90

    blue_x, blue_y = 81.298, 49.511
    blue_yaw = 52.3926 + 90

    green_x, green_y = 161.474, 75.212
    green_yaw = 14.3145 + 90

    # ------------------------------------------------------------
    # Z and stacking parameters
    # ------------------------------------------------------------
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # Use the BLUE block position as the stack anchor (base)
    stack_x = blue_x
    stack_y = blue_y

    # Same orientation for all blocks in final stack: use BLUE yaw
    stack_orient = blue_yaw

    blue_place_z = place_base_z
    red_place_z = place_base_z + block_thickness
    green_place_z = place_base_z + block_thickness * 2.0

    # ------------------------------------------------------------
    # Pick BLUE (base)
    # ------------------------------------------------------------
    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))

    ctx.close_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # ------------------------------------------------------------
    # Place BLUE at base
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, stack_orient))

    ctx.open_gripper("blue")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick RED (middle)
    # ------------------------------------------------------------
    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))

    ctx.close_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # ------------------------------------------------------------
    # Place RED on BLUE
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, stack_orient))

    ctx.open_gripper("red")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick GREEN (top)
    # ------------------------------------------------------------
    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, pick_z, green_yaw))

    ctx.close_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))

    # ------------------------------------------------------------
    # Place GREEN on RED
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, green_place_z, stack_orient))

    ctx.open_gripper("green")

    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Return home
    # ------------------------------------------------------------
    ctx.go_home()
