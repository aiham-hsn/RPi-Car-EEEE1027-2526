from picamera2 import Picamera2
from numpy.typing import NDArray
import numpy as np
import libcamera
import cv2
import time


def process_frame(input_frame: NDArray) -> tuple[NDArray, NDArray]:
    # Convert input frame to grayscale
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (5, 5), 0)

    # Apply an adaptive threshold to convert the image to
    # pure black and pure white
    thresh = cv2.adaptiveThreshold(processed_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 5)

    # Apply Otsu's Binarization to normal thresholding
    # _, thresh = cv2.threshold(processed_gray, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)  # invert colours for morphology operations
    kernel = np.ones((7, 7), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.bitwise_not(thresh)  # make the colours normal again

    return processed_gray, thresh


picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={
    "format": "RGB888",
    "size": (640, 480)
    },
    transform=libcamera.Transform(hflip=1, vflip=1))  # type: ignore
picam2.configure(config)
picam2.start()
time.sleep(2)

print("Processing live feed. Press 'q' in the terminal window to quit.")

try:
    while True:
        # Capture a still frame from the camera
        frame = picam2.capture_array()

        # Process frame using function
        processed, adapt_thresh = process_frame(frame)

        # Display the different frames
        cv2.imshow('Original', frame)
        cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        cv2.imshow('Thresholded', adapt_thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
