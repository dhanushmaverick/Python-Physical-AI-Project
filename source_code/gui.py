import subprocess
import sys
import time


RUN_COMMANDS = {
    "1": {
        "name": "Test camera",
        "cmd": [sys.executable, "-m", "source_code.vision.camera.test_camera"],
    },
    "2": {
        "name": "Capture calibration images",
        "cmd": [sys.executable, "-m", "source_code.vision.calibration.run_capture_images"],
    },
    "3": {
        "name": "Run calibration",
        "cmd": [sys.executable, "-m", "source_code.vision.calibration.run_calibration"],
    },
    "4": {
        "name": "Run undistortion",
        "cmd": [sys.executable, "-m", "source_code.vision.undistortion.run_undistort"],
    },
    "5": {
        "name": "Run homography",
        "cmd": [sys.executable, "-m", "source_code.vision.homography.run_homography"],
    },
    "6": {
        "name": "RoboDK Initialization",
        "cmd": [sys.executable, "-m", "source_code.RoboDK_pran.run_AI_script"],
    },
    "7": {
        "name": "Object segmentation",
        "cmd": [sys.executable, "-m", "source_code.vision.object_segmentation.object_segmentation"],
    },
    "8": {
        "name": "Image to world transformation",
        "cmd": [sys.executable, "-m", "source_code.vision.homography.run_image_to_world_transformation"],
    },
}


def ask_yes_no(question):
    while True:
        answer = input(question + " (yes/no): ").strip().lower()

        if answer in ["yes", "y"]:
            return True

        if answer in ["no", "n"]:
            return False

        print("Please type yes or no.")


def run_command(choice):
    command = RUN_COMMANDS[choice]

    print("\n" + "=" * 40)
    print(f"Starting: {command['name']}")
    print("=" * 40)

    result = subprocess.run(
        command["cmd"],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print(f"\nFinished: {command['name']}")
    else:
        print(f"\nSomething went wrong while running: {command['name']}")

    print("-" * 40)
    time.sleep(1)

    return result


def run_steps(steps):
    for step in steps:
        result = run_command(step)

        if result.returncode != 0:
            print("\nStopped because one step failed.")
            return False

    return True


def output_contains(result, text):
    output_text = (result.stdout or "") + (result.stderr or "")
    return text in output_text


def run_camera_setup():
    """
    If camera/workspace changed:
    1. Try calibration first.
    2. If not enough valid images, capture images.
    3. Then run calibration again.
    4. Then run undistortion and homography.
    """

    calibration_result = run_command("3")

    if calibration_result.returncode != 0:
        if output_contains(calibration_result, "Not enough valid calibration images"):
            print("\nNot enough good calibration images were found.")
            print("I will take new calibration images now.")

            capture_result = run_command("2")

            if capture_result.returncode != 0:
                if output_contains(capture_result, "NO_CAMERA_FOUND"):
                    print("\nNo camera was found.")
                    print("I will now run the camera test.")

                    run_command("1")

                    print("\nPlease fix the camera and run this again.")
                    return False

                print("\nCould not capture calibration images.")
                return False

            print("\nTrying calibration again with the new images.")

            calibration_result = run_command("3")

            if calibration_result.returncode != 0:
                print("\nCalibration still failed.")
                return False

        else:
            print("\nCalibration failed.")
            return False

    return run_steps(["4", "5"])


def simple_automatic_setup():
    print("\n========== SIMPLE AUTOMATIC SETUP ==========")

    new_camera_or_moved = ask_yes_no(
        "Is this a new camera, or was the camera/workspace moved?"
    )

    if new_camera_or_moved:
        print("\nOkay. I will set up the camera again.")

        if not run_camera_setup():
            return
    else:
        print("\nOkay. I will use the existing camera setup.")

    objects_ready = ask_yes_no(
        "\nAre the objects placed on the workspace?"
    )

    if not objects_ready:
        print("\nPlease place the objects on the workspace first.")
        print("Then run this again.")
        return

    print("\nOkay. I will now find the objects and calculate their positions.")

    if not run_steps(["7", "8"]):
        return

    start_robot = ask_yes_no(
        "\nDo you want to start the robot now?"
    )

    if start_robot:
        run_command("6")
    else:
        print("\nRobot start skipped.")


def main():
    while True:
        print("\n========== AUTOMATED MAIN ==========")

        print("w. Simple automatic setup")

        for key, value in RUN_COMMANDS.items():
            print(f"{key}. {value['name']}")

        print("0. Exit")

        choice = input("\nChoose an option: ").strip().lower()

        if choice == "0":
            print("Exited")
            break

        if choice == "w":
            simple_automatic_setup()

        elif choice in RUN_COMMANDS:
            run_command(choice)

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()