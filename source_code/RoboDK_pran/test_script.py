def run_task(robot):
    """
    Test AI-generated motion plan.

    Task:
    Stack red on green, then blue on top.

    Final stack:
    bottom = green
    middle = red
    top = blue
    """

    # -----------------------------
    # Block poses from vision system
    # -----------------------------
    red_x = 19.924
    red_y = 9.888
    red_yaw = 89.7196

    green_x = 161.474
    green_y = 75.212
    green_yaw = 26.9757

    blue_x = 81.298
    blue_y = 49.511
    blue_yaw = 47.3715

    # -----------------------------
    # Update visible block positions
    # -----------------------------
    robot.update_block_pose("red", red_x, red_y, 0.0, red_yaw)
    robot.update_block_pose("green", green_x, green_y, 0.0, green_yaw)
    robot.update_block_pose("blue", blue_x, blue_y, 0.0, blue_yaw)

    # -----------------------------
    # Stack settings
    # -----------------------------
    stack_x = green_x
    stack_y = green_y
    block_thickness = 20.0

    green_place_z = 0.0
    red_place_z = block_thickness
    blue_place_z = block_thickness * 2.0

    # -----------------------------
    # Pick green and place as bottom
    # -----------------------------
    robot.pick("green", green_x, green_y, green_yaw)
    robot.place(stack_x, stack_y, green_place_z, green_yaw)

    # -----------------------------
    # Pick red and place on green
    # -----------------------------
    robot.pick("red", red_x, red_y, red_yaw)
    robot.place(stack_x, stack_y, red_place_z, red_yaw)

    # -----------------------------
    # Pick blue and place on red
    # -----------------------------
    robot.pick("blue", blue_x, blue_y, blue_yaw)
    robot.place(stack_x, stack_y, blue_place_z, blue_yaw)

    # -----------------------------
    # Return robot to home
    # -----------------------------
    robot.go_home()