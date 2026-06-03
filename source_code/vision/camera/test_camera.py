# Run command:
# python -m source_code.vision.camera.test_camera
# python -m source_code.vision.camera.test_camera --camera 1 # (if you have multiple cameras and want to test a different one)

import argparse
import cv2

from source_code.vision.camera.camera import Webcam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        type=int,
        default=2,
        help="Camera number to use. Default is 0."
    )

    args = parser.parse_args()

    webcam = Webcam(camera_index=args.camera, width=1280, height=720)

    try:
        webcam.open()
        print(f"Webcam {args.camera} opened successfully.")
        print("Press Q to quit.")

        while True:
            frame = webcam.read()
            cv2.imshow(f"Webcam Test - Camera {args.camera}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()