import cv2


class Webcam:
    def __init__(self, camera_index: int = 0  #has to be editd if different webcam is used
                 , width: int | None = None, height: int | None = None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Webcam is not opened")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None