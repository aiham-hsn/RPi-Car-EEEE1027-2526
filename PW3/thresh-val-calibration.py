from typing import Union
from numpy.typing import NDArray
from picamera2 import Picamera2
import numpy as np
import libcamera
import cv2
import time
from datetime import UTC, datetime as dt


def process_frame_otsu(
    input_frame: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], Union[int, float]]:
    # Convert input frame to grayscale
    # processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (7, 7), 0)

    # Apply Otsu's Binarization to normal thresholding
    computed_thres_val, thresh = cv2.threshold(
        processed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return processed_gray, thresh, computed_thres_val  # pyright: ignore[reportReturnType]


cam_size_x = 640
cam_size_y = 480

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={
    "format": "RGB888",
    "size": (cam_size_x, cam_size_y)
    },
    transform=libcamera.Transform(hflip=1, vflip=1))  # type: ignore
picam2.configure(config)
picam2.start()
time.sleep(2)

threshval = -1

try:
    # Capture a still frame from the camera
    frame = picam2.capture_array()

    # Process frame using Otsu function and output threshold value returned by Otsu to STDOUT
    processed, thresh, threshval = process_frame_otsu(frame)
    if threshval != -1:
        print(f"\nComputed Thresh Val: [{threshval}]")
        unix_timestamp = int(dt.now(UTC).timestamp())
        cv2.imwrite(f'./threshval-calib-outframe-{unix_timestamp}.jpg', thresh)

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
