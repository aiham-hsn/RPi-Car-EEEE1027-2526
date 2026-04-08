from typing import Union
from numpy.typing import NDArray
from gpiozero import Motor, PWMOutputDevice
from picamera2 import Picamera2
import numpy as np
import libcamera
import cv2
import time
from math import copysign


def process_frame(
    input_frame: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # Convert input frame to HSV colour space for colour line operations

    # Convert input frame to grayscale
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (9, 9), 0)

    # Just use normal thresholding
    _, thresh = cv2.threshold(processed_gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Apply morphology operations
    thresh = process_frame_morphology_ops(thresh)

    return processed_gray, thresh  # pyright: ignore[reportReturnType]


def process_frame_morphology_ops(
        input_thresh: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # Create kernel for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    out_thresh = cv2.erode(input_thresh, kernel, iterations=1)

    return out_thresh


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


def detect_colored_line(raw_frame, colour):
    hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
    if colour == "black":
        return False, None
    elif colour == "yellow":
        ylw_low, ylw_high = COLOUR_RANGES['yellow']
        ylw_mask = cv2.inRange(hsv, np.array(ylw_low), np.array(ylw_high))

        mask = process_frame_morphology_ops(ylw_mask)
        colour_present = np.count_nonzero(mask) > 0
        return colour_present, mask
    elif colour == "red":
        red_toplow, red_tophigh = COLOUR_RANGES['red']['top']
        red_btmlow, red_btmhigh = COLOUR_RANGES['red']['btm']
        red_masktop = cv2.inRange(hsv, np.array(red_toplow),
            np.array(red_tophigh))
        red_maskbtm = cv2.inRange(hsv, np.array(red_btmlow),
            np.array(red_btmhigh))
        red_mask = cv2.bitwise_or(red_maskbtm, red_masktop)

        mask = process_frame_morphology_ops(red_mask)
        colour_present = np.count_nonzero(mask) > 0
        return colour_present, mask
    elif colour == "both":
        ylw_low, ylw_high = COLOUR_RANGES['yellow']
        ylw_mask = cv2.inRange(hsv, np.array(ylw_low), np.array(ylw_high))

        red_toplow, red_tophigh = COLOUR_RANGES['red']['top']
        red_btmlow, red_btmhigh = COLOUR_RANGES['red']['btm']
        red_masktop = cv2.inRange(hsv, np.array(red_toplow),
            np.array(red_tophigh))
        red_maskbtm = cv2.inRange(hsv, np.array(red_btmlow),
            np.array(red_btmhigh))
        red_mask = cv2.bitwise_or(red_maskbtm, red_masktop)

        mask = cv2.bitwise_or(ylw_mask, red_mask)

        mask = process_frame_morphology_ops(red_mask)
        colour_present = np.count_nonzero(mask) > 0
        return colour_present, mask
    else:
        raise ValueError(f"Unknown value for input value \"colour\" [{colour}]")


def linefollowing_colourchoice() -> str:
    options = {
        '1': ('black', 'Black line only'),
        '2': ('red', 'Red line or black line'),
        '3': ('yellow', 'Yellow line or black line'),
        '4': ('both', 'Red or yellow or black line'),
    }
    print("Select line colour(s) to follow.\nOptions:")
    for key, (_, desc) in options.items():
        print(f"{key} : {desc}")

    while True:
        raw = input("\nEnter choice [1-4]: ").strip()
        if raw in options:
            color, desc = options[raw]
            print(f"Line following colour choice set to: [{desc}]\n")
            return color
        print(f"  Invalid choice {raw!r} — please enter 1, 2, 3, or 4.")


cam_size_x = 640
cam_size_y = 480

global COLOUR_RANGES, PRIORITY_COLOUR
COLOUR_RANGES = {
    "red": {
    "top": [(170, 150, 100), (180, 255, 255)],
    "btm": [(0, 150, 100), (10, 255, 255)],
    },
    "yellow": [(20, 190, 190), (30, 255, 255)],
}
PRIORITY_COLOUR = linefollowing_colourchoice()

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={
    # the RGB888 format has 24 bits per pixel, ordered [B, G, R]
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

try:
    while True:
        # Capture a still frame from the camera
        frame = picam2.capture_array()

        # Process frame using function
        processed, thresh = process_frame(frame)
        # proc = process_frame_alt(frame)

        height, width = np.shape(thresh)

        frame_roi = frame[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height *
            (1 - frame_discard_offset)):]
        frame_roi_w_contour = frame_roi.copy()
        thresh_roi = thresh[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height *
            (1 - frame_discard_offset)):]

        # Check for colour in the ROI
        # colour_present, colour_mask = detect_colored_line(frame_roi, 'both')
        colour_present, colour_mask = detect_colored_line(frame_roi, PRIORITY_COLOUR)
        if colour_present:
            thresh_roi = cv2.bitwise_and(thresh_roi, colour_mask)
        print(f"Colour present? : [{colour_present}]")

        main_contour = thresh2maincontour(thresh_roi)

        if main_contour is not None:
            cv2.drawContours(frame_roi_w_contour, [main_contour], 0,
                (0, 255, 0), 3)

        # Display the different frames
        # cv2.imshow('Original', frame)
        # cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        # cv2.imshow('Thresholded', thresh)
        cv2.imshow('Orignal ROI', frame_roi)
        cv2.imshow('Thresholded ROI', thresh_roi)
        cv2.imshow(
            'ROI w/ contours',
            frame_roi_w_contour  # pyright: ignore[reportPossiblyUnboundVariable]
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
