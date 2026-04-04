import cv2
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.vision.webcam import Webcam


def main():
    webcam = Webcam(camera_index=0, width=1280, height=720)

    try:
        webcam.open()
        print("Webcam opened successfully.")
        print("Press Q to quit.")

        while True:
            frame = webcam.read()
            cv2.imshow("Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()