from typing import Union
from numpy.typing import NDArray
from picamera2 import Picamera2
import numpy as np
import libcamera
import cv2
import time
from math import copysign


def process_frame(
    input_frame: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # Convert input frame to grayscale
    # processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (9, 9), 0)

    # Just use normal thresholding
    _, thresh = cv2.threshold(processed_gray, 160, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return processed_gray, thresh  # pyright: ignore[reportReturnType]


def process_frame_alt(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary)
    kern = np.ones((5, 5), np.uint8)
    er = cv2.erode(inv, kern, iterations=1)
    return cv2.bitwise_not(er)


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


def thresh2maincontour(threshhold):
    # Get contours from threshhold
    contours, hierarchy = cv2.findContours(threshhold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Find and return main contour
    ## modified from https://github.com/tprlab/pitanq-dev
    largest_contour = None
    if contours is not None and len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
    if largest_contour is None:
        return None
    return largest_contour


def calc_centroid(main_cnt):
    moments = cv2.moments(main_cnt)
    centroid_x = int(moments['m10'] / (moments['m00'] or 1))
    centroid_y = int(moments['m01'] / (moments['m00'] or 1))
    return centroid_x, centroid_y


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

print("Processing live feed. Press 'q' in the terminal window to quit.")

# %age of the top half of the frame to discard to get the ROI
frame_discard_percentage = 0.3
# %age by which the ROI is being moved upwards
frame_discard_offset = 0.125

threshval = -1

start_time = time.time()
last_time = time.time()
try:
    while True:
        # Capture a still frame from the camera
        frame = picam2.capture_array()

        now = time.time()
        dt = now - last_time
        last_time = now

        # Process frame using function
        processed, thresh = process_frame(frame)
        # proc = process_frame_alt(frame)
        #processed, thresh, threshval = process_frame_otsu(frame)
        #print(f"Computed Thresh Val: [{threshval}]")

        height, width = np.shape(thresh)

        frame_roi = frame[int(height *
                              (frame_discard_percentage - frame_discard_offset)
                              ):int(height * (1 - frame_discard_offset)):]
        frame_roi_w_contours = frame_roi.copy()
        thresh_roi = thresh[int(height * (frame_discard_percentage -
                                          frame_discard_offset)):int(height * (
                                              1 - frame_discard_offset)):]

        contours, hierarchy = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours onto frame ROI
        cv2.drawContours(frame_roi_w_contours, contours, -1, (0, 255, 0), 3)

        # Check areas of the contours
        #for idx, cnt in enumerate(contours):
        #    area = cv2.contourArea(cnt)
        #    print(f'Contour {idx} Area : {area}')
        detected_arrow = 'None'
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1000 < area < 15000:
                hull = cv2.convexHull(cnt)
                solidity = float(area) / cv2.contourArea(
                    hull) if cv2.contourArea(hull) > 0 else 0
                if 0.52 <= solidity <= 0.75:
                    x, y, w, h = cv2.boundingRect(cnt)
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] /
                                                               M["m00"])
                        bX, bY = x + (w / 2), y + (h / 2)
                        if abs(cX - bX) > abs(cY - bY):
                            detected_arrow = "Arrow Right" if cX > bX else "Arrow Left"
                        else:
                            detected_arrow = "Arrow Down" if cY > bY else "Arrow Up"
                        break
                    else:
                        detected_arrow = 'None'
                else:
                    detected_arrow = 'None'
            else:
                detected_arrow = 'None'
        print(f'Currently Detected Arrow : {detected_arrow}')

        # Display the different frames
        # cv2.imshow('Original', frame)
        # cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        # cv2.imshow('Thresholded', thresh)
        # cv2.imshow('Orignal ROI', frame_roi)
        # cv2.imshow('Thresholded ROI', thresh_roi)
        # cv2.imshow(
        #     'ROI w/ contours',
        #     frame_roi_w_contours # pyright: ignore[reportPossiblyUnboundVariable]
        # )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\n\nTIme Taken : [{time.time() - start_time}]")
