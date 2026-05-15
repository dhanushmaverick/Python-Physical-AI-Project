from app.utils.paths import INTRINSICS_FILE
import cv2
import numpy as np

from app.vision.undistort import load_intrinsics, undistort_frame
from app.vision.homography import compute_homography, image_to_world


clicked_points = []


def mouse_callback(event, x, y, flags, image): # the set MouseCallback function will call this mouse_callback function whenever a mouse event occurs in the specified window, 
                                               # it also gives us the event type, the x and y coordinates of the mouse event, any flags associated with the event, and a user-defined parameter (in this case, the image) that we can use to visualize the clicked points on the image
    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_points.append([x, y])
        point_number = len(clicked_points)

        print(f"Point {point_number}: image coordinate = ({x}, {y})")

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  #green circle to visualize the clicked point on the image, with a radius of 5 pixels and a thickness of -1 (which means the circle will be filled) 1 can be used to outline 

        cv2.putText(    # placing the point number nearby to the clicked point  
            image,
            str(point_number),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, # mandatory parameters for the putText function is to specify the text, its position, font, scale, color, and thickness
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Click 4 reference points", image)


# loading camera intrinsics 

camera_matrix, dist_coeffs = load_intrinsics(INTRINSICS_FILE)

# ------------------------------------------------------

            # CHANGE THE FILE TO IMAGE PATH SUCH THAT YOU KNOW THE REAL-WORLD COORDINATES OF THE 4 POINTS YOU CLICK IN THE IMAGE

IMG_PATH = "data/calibration/raw_images/Im_L_1.png"  # CHANGE THIS TO YOUR IMAGE PATH

img = cv2.imread(IMG_PATH)

# ------------------------------------------------------



if img is None:
    raise FileNotFoundError("Could not read image from: {IMG_PATH} ")  


undistorted = undistort_frame(img, camera_matrix, dist_coeffs)   # calling the undistortion function to get the undistorted image for better accuracy when clicking the reference points for homography computation


display_img = undistorted.copy()

print("Click 4 reference points in the image.")
print("Use the same order as your world_pts.")
print("Example order: top-left, top-right, bottom-right, bottom-left")
print("Press any key after selecting 4 points.")

cv2.imshow("Click 4 reference points", display_img)     # displaying the undistorted image and setting up the mouse callback to capture the clicked points for homography computation
cv2.setMouseCallback("Click 4 reference points", mouse_callback, display_img) #cv2 function to set the mouse callback for the window, which will call any function whenever a mouse event occurs in that window, allowing us to capture the clicked points and visualize them on the image

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:  # edit this condition if needed but for homography computation we typically need exactly 4 corresponding points between the world and image
    raise RuntimeError(f"Expected 4 clicked points, but got {len(clicked_points)}")

image_pts = np.array(clicked_points, dtype=np.float32)  # converting the list of clicked points to a numpy array of type float32, which is the required format for the homography computation 

print("\nClicked image points:")
print(image_pts)

# real-world measured coordinates in $ -- mm -- $
# IMPORTANT: must match the same order as the clicks

#  part to be edited to input the real-world coordinates of the 4 points of the real workspace
#  The order of these points must match the order in which they were clicked in the image for accurate homography computation.
world_pts = np.array([
    [0, 0],
    [300, 0],
    [300, 210],
    [0, 210]
], dtype=np.float32)

print("\nWorld points:")
print(world_pts)

# compute homography
H = compute_homography(world_pts, image_pts)

print("\nHomography matrix:")
print(H)



# test a pixel coordinate (in the image) to see what its corresponding world coordinate is 
# using the computed homography matrix, this is just an example and can be changed to any pixel coordinate you want to test
pixel = np.array([320, 240], dtype=np.float32) 

world = image_to_world(pixel, H)

print("\nTest pixel:", pixel)
print("World coord:", world)


## all the coordinates founded by the Iblob function from Joel's part can be transformed to world coordinates 
# that code will follow once the homography computation is verified to be working correctly with the test pixel coordinate.