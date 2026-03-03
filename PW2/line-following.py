from typing import Union
from numpy.typing import NDArray
from cv2.typing import MatLike
from gpiozero import Motor, PWMOutputDevice
from picamera2 import Picamera2
from simple_pid import PID
import numpy as np
import libcamera
import cv2
import time


def set_duty_cycle_both(input: Union[int, float]) -> None:
    if input < 0:
        left_pwm.value = 0
        right_pwm.value = 0
    elif input > 1:
        left_pwm.value = 1
        right_pwm.value = 1
    else:
        left_pwm.value = input
        right_pwm.value = input


def set_duty_cycle_left(input: Union[int, float]) -> None:
    if input < 0:
        left_pwm.value = 0
    elif input > 1:
        left_pwm.value = 1
    else:
        left_pwm.value = input


def set_duty_cycle_right(input: Union[int, float]) -> None:
    if input < 0:
        right_pwm.value = 0
    elif input > 1:
        right_pwm.value = 1
    else:
        right_pwm.value = input


def drive_fwd(ds: Union[int, float]):
    set_duty_cycle_both(ds)
    left_dir.forward()
    right_dir.forward()


def drive_bckwd(ds: Union[int, float]):
    set_duty_cycle_both(ds)
    left_dir.backward()
    right_dir.backward()


def turn_right(duty_cycle_left: Union[int, float], duty_cycle_right: Union[int,
    float]) -> None:
    set_duty_cycle_left(duty_cycle_left)
    set_duty_cycle_right(duty_cycle_right)
    left_dir.forward()
    right_dir.backward()


def turn_left(duty_cycle_left: Union[int, float], duty_cycle_right: Union[int,
    float]) -> None:
    set_duty_cycle_left(duty_cycle_left)
    set_duty_cycle_right(duty_cycle_right)
    left_dir.backward()
    right_dir.forward()


def stop_car():
    set_duty_cycle_both(0)
    left_dir.stop()
    right_dir.stop()


def process_frame(input_frame: NDArray[np.uint8]) -> tuple[MatLike, MatLike]:
    # Convert input frame to grayscale
    # processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (7, 7), 0)

    # Just use normal thresholding
    _, thresh = cv2.threshold(processed_gray, 145, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7, 7), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return processed_gray, thresh


def process_frame_adaptive(
        input_frame: NDArray[np.uint8]) -> tuple[MatLike, MatLike]:
    # Convert input frame to grayscale
    # processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (7, 7), 0)

    # Apply an adaptive threshold to convert the image to
    # pure black and pure white
    thresh = cv2.adaptiveThreshold(processed_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)

    kernel = np.ones((7, 7), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return processed_gray, thresh


def process_frame_otsu(
    input_frame: NDArray[np.uint8]
) -> tuple[MatLike, MatLike, Union[int, float]]:
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

    return processed_gray, thresh, computed_thres_val


def find_main_countour(input_contours):
    ## modified from https://github.com/tprlab/pitanq-dev
    largest_contour = None
    if input_contours is not None and len(input_contours) > 0:
        largest_contour = max(input_contours, key=cv2.contourArea)
    if largest_contour is None:
        return None
    return largest_contour


cam_size_x = 640
cam_size_y = 480

ENA = 13  # Control right side motors; GPIO/BCM pin 13, Physical/Board pin 33
ENB = 19  # Control left side motors;  GPIO/BCM pin 19, Physical/Board pin 35

IN1 = 'BCM24'  # Controls the IN1 input on the L298N; GPIO/BCM pin 24, Physical/Board pin 18
IN2 = 'BCM23'  # Controls the IN2 input on the L298N; GPIO/BCM pin 23, Physical/Board pin 16
IN3 = 'BCM27'  # Controls the IN3 input on the L298N; GPIO/BCM pin 27, Physical/Board pin 13
IN4 = 'BCM22'  # Controls the IN4 input on the L298N; GPIO/BCM pin 22, Physical/Board pin 15

# Init gpiozero Motors
left_dir = Motor(forward=IN1, backward=IN2)
right_dir = Motor(forward=IN3, backward=IN4)

# Init PWM control of motors
left_pwm = PWMOutputDevice(ENA, frequency=1000)
right_pwm = PWMOutputDevice(ENB, frequency=1000)

pid = PID(
    Kp=0.8,
    Ki=0.03,
    Kd=0.35,
    setpoint=(cam_size_x / 2),
    sample_time=0.01,
    output_limits=(0, 40),
    starting_output=(cam_size_x / 2))

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
frame_discard_percentage = 0.4
# %age by which the ROI is being moved upwards
frame_discard_offset = 0.125

threshval = -1
BASE_SPEED = 0.65

try:
    while True:
        # Capture a still frame from the camera
        frame = picam2.capture_array()

        # Process frame using function
        processed, thresh = process_frame(frame)
        # processed, thresh, threshval = process_frame_otsu(frame)
        # print(threshval)

        height, width = np.shape(thresh)

        frame_roi = frame[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height *
            (1 - frame_discard_offset)):]
        frame_roi_w_contours = frame_roi
        thresh_roi = thresh[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height *
            (1 - frame_discard_offset)):]

        contours, hierarchy = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        main_contour = find_main_countour(contours)

        if main_contour is not None:
            moments = cv2.moments(main_contour)
            centroid_x = int(moments['m10'] / (moments['m00'] or 1))
            centroid_y = int(moments['m01'] / (moments['m00'] or 1))

            correction = pid(centroid_x)
            print(f"Line center: [{centroid_x}] | PID: [{correction}] | ")

            frame_roi_w_contours = cv2.drawContours(frame_roi, main_contour, -1,
                (0, 255, 0), 3)
            cv2.circle(frame_roi_w_contours, (centroid_x, centroid_y), 4,
                (255, 0, 0), 4)

        # Display the different frames
        # cv2.imshow('Original', frame)
        # cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        cv2.imshow('Thresholded', thresh)
        # cv2.imshow('Orignal ROI', frame_roi)
        # cv2.imshow('Thresholded ROI', thresh_roi)
        cv2.imshow('ROI w/ contours', frame_roi_w_contours)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    stop_car()
    picam2.stop()
    cv2.destroyAllWindows()
