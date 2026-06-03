def run_motion(ctx):
    """Stacking plan based on current block coordinates.

    User request:
    - Place RED block on the GREEN block
    - Place BLUE block on the RED block

    Note: Add +90 to the true yaw angles of the blocks in this generated code
    to align the gripper laterally for gripping.
    """

    robot = ctx.robot

    # ------------------------------------------------------------
    # World coordinates of blocks (from get_world_coords_string)
    # ------------------------------------------------------------
    red_x, red_y = 82.791, 56.399
    red_yaw = 55.0003 + 90

    green_x, green_y = 21.822, -22.004
    green_yaw = 149.8728 + 90

    blue_x, blue_y = -13.848, 72.704
    blue_yaw = 81.0174 + 90

    # ------------------------------------------------------------
    # Z heights
    # ------------------------------------------------------------
    approach_z = ctx.approach_z
    pick_z = ctx.pick_z
    place_base_z = ctx.place_base_z
    block_thickness = ctx.block_thickness

    # Stack at the GREEN x/y (bottom)
    stack_x = green_x
    stack_y = green_y

    green_place_z = place_base_z
    red_place_z = place_base_z + block_thickness
    blue_place_z = place_base_z + block_thickness * 2.0

    # Use GREEN orientation for all place steps so alignment is consistent
    stack_orient = green_yaw

    # ------------------------------------------------------------
    # Pick GREEN
    # ------------------------------------------------------------
    ctx.open_gripper("green")
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, pick_z, green_yaw))
    ctx.close_gripper("green")
    robot.MoveL(ctx.pose_xyz_yaw(green_x, green_y, approach_z, green_yaw))

    # ------------------------------------------------------------
    # Place GREEN (bottom)
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, green_place_z, stack_orient))
    ctx.open_gripper("green")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick RED
    # ------------------------------------------------------------
    ctx.open_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, pick_z, red_yaw))
    ctx.close_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(red_x, red_y, approach_z, red_yaw))

    # ------------------------------------------------------------
    # Place RED on GREEN
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, red_place_z, stack_orient))
    ctx.open_gripper("red")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # ------------------------------------------------------------
    # Pick BLUE
    # ------------------------------------------------------------
    ctx.open_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, pick_z, blue_yaw))
    ctx.close_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(blue_x, blue_y, approach_z, blue_yaw))

    # ------------------------------------------------------------
    # Place BLUE on RED
    # ------------------------------------------------------------
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, blue_place_z, stack_orient))
    ctx.open_gripper("blue")
    robot.MoveL(ctx.pose_xyz_yaw(stack_x, stack_y, approach_z, stack_orient))

    # Return home
    ctx.go_home()
