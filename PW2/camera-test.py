from picamera2 import Picamera2
import libcamera
import cv2
import numpy as np
import time

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
        # 1. Capture the frame as an array
        frame = picam2.capture_array()

        # 2. Grayscale Conversion: Simplifies processing [cite: 102]
        # Note: Picamera2 RGB888 needs RGB2GRAY, not BGR2GRAY
        processed = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 3. Noise Reduction: Gaussian filtering [cite: 103]
        # This removes tiny details like floor scratches [cite: 104]
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

        adapt_thresh = cv2.adaptiveThreshold(processed, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

        # 4. Display the results
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
