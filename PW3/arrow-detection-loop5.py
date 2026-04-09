from typing import Union
from numpy.typing import NDArray
from picamera2 import Picamera2
from collections import deque, Counter
import numpy as np
import libcamera
import cv2
import time


def process_frame(input_frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # Convert input frame to grayscale
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (9, 9), 0)

    # Just use normal thresholding
    _, thresh = cv2.threshold(processed_gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Apply morphology operations
    thresh = process_frame_morphology_ops(thresh)

    return processed_gray, thresh  # pyright: ignore[reportReturnType]


def process_frame_morphology_ops(input_thresh):
    # Create kernel for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    out_thresh = cv2.erode(input_thresh, kernel, iterations=1)

    return out_thresh


def thresh2maincontour(threshhold):
    # Get contours from threshhold
    contours, _ = cv2.findContours(threshhold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def list_getMode(input_list: list) -> Union[str, None]:
    return Counter(input_list).most_common(1)[0][0]


cam_size_x = 640
cam_size_y = 480

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={
    "format": "RGB888",
    "size": (cam_size_x, cam_size_y)
    }, transform=libcamera.Transform(hflip=1, vflip=1))  # type: ignore
picam2.configure(config)
picam2.start()
time.sleep(2)

print("Processing live feed. Press 'q' in the terminal window to quit.")

# %age of the top half of the frame to discard to get the ROI
frame_discard_percentage = 0.3
# %age by which the ROI is being moved upwards
frame_discard_offset = 0.125

BLACK_RANGE = [(0, 0, 0), (180, 160, 110)]

arrow_detections = deque([])

global detected_arrow
detected_arrow = None

try:
    for loop_idx in range(0, 5):
        # Capture a still frame from the camera
        frame = picam2.capture_array()

        # Process frame using function
        processed, thresh = process_frame(frame)

        height, width = np.shape(thresh)

        frame_roi = frame[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height * (1 - frame_discard_offset)):]
        thresh_roi = thresh[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height * (1 - frame_discard_offset)):]

        # Get mask of all black pixels in the image
        black_mask = cv2.inRange(
            cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV), np.array(BLACK_RANGE[0]), np.array(BLACK_RANGE[1]))
        thresh_roi = cv2.bitwise_xor(thresh_roi, black_mask)

        thresh_roi = process_frame_morphology_ops(thresh_roi)

        contours, hierarchy = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the arrow
        # Assumption is only possible due to the fact that the black line is being filtered out
        arrow_cnt = max(contours, key=cv2.contourArea)
        detected_arrow = None
        arrow_area = 0

        x, y, w, h = cv2.boundingRect(arrow_cnt)
        arrow_cnt_moments = cv2.moments(arrow_cnt)
        if arrow_cnt_moments["m00"] > 0:
            # Get centroid of arrow contour
            cX, cY = int(arrow_cnt_moments["m10"] / arrow_cnt_moments["m00"]), int(arrow_cnt_moments["m01"] / arrow_cnt_moments["m00"])

            # Get the center of the bounding box of the arrow contour
            bX, bY = x + (w / 2), y + (h / 2)

            if abs(cX - bX) > abs(cY - bY):
                detected_arrow = "Arrow Right" if cX > bX else "Arrow Left"
                arrow_area = cv2.contourArea(arrow_cnt)

            else:
                detected_arrow = "Arrow Down" if cY > bY else "Arrow Up"
                arrow_area = cv2.contourArea(arrow_cnt)
        else:
            detected_arrow = None
        arrow_detections.appendleft(detected_arrow)
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    most_common_arrow_detection = list_getMode(list(arrow_detections))
    if most_common_arrow_detection is None:
        print('[ARROW DETECTION] Arrow symbol not reliably detected')
    else:
        print(f'[ARROW DETECTION] Most commonly detected arrow: {most_common_arrow_detection}')
