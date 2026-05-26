# main.py

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
        "cmd": [sys.executable, "-m", "source_code.vision.Robodk.robodk_init"],
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





def main():
    while True:
        print("\n========== AUTOMATED MAIN ==========")

        for key, value in RUN_COMMANDS.items():
            print(f"{key}. {value['name']}")

        print("0. Exit")

        choice = input("\nChoose an option: ")

        if choice == "0":
            break

        if choice in RUN_COMMANDS:
            run_command(choice)
            
        else:
            print("Invalid choice.")

def run_command(choice):
    command = RUN_COMMANDS[choice]

    print(f"\nRunning: {command['name']}")
    print("Command:", " ".join(command["cmd"]))

    subprocess.run(command["cmd"])
    print("-"*20,"\n\n" ,f"Finished Running: {command['name']}")
    time.sleep(3)



if __name__ == "__main__":
    main()