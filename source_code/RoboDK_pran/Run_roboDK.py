#run command :
#       python -m source_code.RoboDK_pran.Run_roboDK

from robodk.robolink import Robolink, ITEM_TYPE_ROBOT

from source_code.utility.paths import ROBO_DK_STATION_PATH


ROBOT_NAME = "UR5e"
HOME_TARGET_NAME = "HOME"

BLOCK_FRAMES = {
    "red": "WORLD_FRAME",
    "green": "WORLD_FRAME",
    "blue": "WORLD_FRAME",
}

APPROACH_Z = 80.0
PICK_Z = 5.0


def main():
    print("[INFO] Connecting to RoboDK...")

    rdk = Robolink() # Connects to the RoboDK API and starts the RoboDK application if it is not already running.
    # RoboDK should be present in the local machine under the default installation path. If RoboDK is not found, an error will be raised.
    #if the default installation path is not used, the path to RoboDK should be added to the system PATH variable and the RoboDK API should be installed via pip.
    # USE WITH CAUTION : rdk = Robolink(robodk_path="C:/bin/RoboDK/bin/RoboDK.exe") 

    print("[INFO] Opening RoboDK station:")
    print(ROBO_DK_STATION_PATH) 

    rdk.AddFile(str(ROBO_DK_STATION_PATH)) #this function opens the specified RoboDK station file. If the file is not found, an error will be raised.

    print(f"[INFO] Searching for robot: {ROBOT_NAME}")
    print(ITEM_TYPE_ROBOT)
    robot = rdk.Item(ROBOT_NAME, ITEM_TYPE_ROBOT) #each item in the RoboDK station has a type and a name. This function searches for an item with the specified name and type. 
    # If the item is not found, an error will be raised.

    if not robot.Valid():
        raise RuntimeError(
            f"Robot '{ROBOT_NAME}' was not found in the RoboDK station.\n"
            "Check the exact robot name in the RoboDK station tree."
        )

    print("[SUCCESS] Connected to RoboDK.")
    print(f"[SUCCESS] Robot found: {robot.Name()}")
    print("Close the Window to end the program.")


if __name__ == "__main__":
    main()