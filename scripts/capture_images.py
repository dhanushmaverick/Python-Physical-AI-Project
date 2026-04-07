from app.utils.paths import RAW_IMAGES_DIR, ensure_directories
import cv2

ensure_directories()

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = RAW_IMAGES_DIR / f"img_{count:02d}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()