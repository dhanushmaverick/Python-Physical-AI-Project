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
        "name": "RoboDK Simulation",
        "cmd": [sys.executable, "-m", "source_code.simulation.run_AI_script"],
    },
    "7": {
        "name": "Object segmentation",
        "cmd": [sys.executable, "-m", "source_code.vision.object_segmentation.object_segmentation"],
    },
    "8": {
        "name": "Image to world transformation",
        "cmd": [sys.executable, "-m", "source_code.vision.homography.run_image_to_world_transformation"],
    },
    "9": {
        "name": "Clear existing calibration images",
        "cmd": [sys.executable, "-m", "source_code.vision.camera.clear_raw_images"],
    },
}


def ask_yes_no(question):
    while True:
        answer = input(question + " (yes/no or q to quit): ").strip().lower()
        if answer in ["yes", "y"]:return True
        if answer in ["no", "n"]:return False
        if answer == "q":
            print("\nExiting program.")
            sys.exit(0)
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


def output_contains(result, text):
    output_text = (result.stdout or "") + (result.stderr or "")
    return text in output_text


def run_new_camera_setup():
    """
    New camera means:
    optional clear images -> capture images -> calibration -> undistortion -> homography

    Test camera runs only if capture images gives NO_CAMERA_FOUND.
    """

    print("\nOkay. I will set up the new camera.")

    clear_images = ask_yes_no("Do you want to clear existing calibration images?")

    if clear_images:
        clear_result = run_command("9")

        if clear_result.returncode != 0:
            print("\nCould not clear existing calibration images.")
            return False
    else:
        print("\nOkay. I will keep the existing calibration images.")

    capture_result = run_command("2")

    if capture_result.returncode != 0:
        if output_contains(capture_result, "NO_CAMERA_FOUND"):
            print("\nNo camera was found.")
            print("I will now run the camera test.")

            run_command("1")

            print("\nPlease fix the camera and run this setup again.")
            return False

        print("\nCould not capture calibration images.")
        return False

    return run_multiple_cmds(["3", "4", "5"])


def run_workspace_moved_setup():
    """
    If the same camera or workspace was moved,
    only homography is needed.
    """

    print("\nOkay. I will update the workspace position.")
    return run_command("5").returncode == 0


def ask_object_and_simulation_choice():
    print("\nWhat do you want to do now?")
    print("1. Find objects first, then Prompt the task?")
    print("2. Directly Prompt the task?")

    while True:
        choice = input("\nChoose 1, 2, or q to quit: ").strip().lower()

        if choice == "q":
            print("\nQuitting program.")
            sys.exit(0)

        if choice == "1":
            print("\nOkay. I will find the objects first.")

            if not run_multiple_cmds(["7", "8"]):
                return

            print("\nNow I will start the simulation.")
            run_command("6")
            return

        if choice == "2":
            print("\nOkay. I will directly start the simulation.")
            run_command("6")
            return

        print("Please choose 1, 2, or q.")

def run_multiple_cmds(steps):
    for step in steps:
        result = run_command(step)

        if result.returncode != 0:
            print("\nStopped because one step failed.")
            return False

    return True


def simple_automatic_setup():
    print("\n========== PHYSICAL AI ==========")

    new_camera = ask_yes_no("Is this a new camera?")


    if new_camera:
        if not run_new_camera_setup():
            return

    else:
        moved = ask_yes_no("Was the camera or workspace moved?")

        if moved:
            if not run_workspace_moved_setup():
                return
        else:
            print("\nOkay. No camera setup is needed.")

    ask_object_and_simulation_choice()

if __name__ == "__main__":
    simple_automatic_setup()