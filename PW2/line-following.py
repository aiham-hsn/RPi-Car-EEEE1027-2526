from typing import Union
from gpiozero import Motor, PWMOutputDevice
from numpy.typing import NDArray
from picamera2 import Picamera2
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


def process_frame(input_frame: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    # Apply CLAHE on V channel to inprove contrast
    h, s, v = cv2.split(cv2.cvtColor(input_frame, cv2.COLOR_RGB2HSV))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    frame_eq = cv2.cvtColor(
        cv2.merge([h, s, clahe.apply(v)]), cv2.COLOR_HSV2RGB)

    # Convert input frame to grayscale
    # processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(frame_eq, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (7, 7), 0)

    # Apply an adaptive threshold to convert the image to
    # pure black and pure white
    thresh = cv2.adaptiveThreshold(processed_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)

    # Apply Otsu's Binarization to normal thresholding
    # _, thresh = cv2.threshold(processed_gray, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresh = cv2.bitwise_not(thresh)  # invert colours for morphology operations
    kernel = np.ones((7, 7), np.uint8)  # for morphology operations
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.bitwise_not(thresh)  # make the colours normal again

    return frame_eq, processed_gray, thresh


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
        frame_eq, processed, adapt_thresh = process_frame(frame)

        # Display the different frames
        cv2.imshow('Original', frame)
        cv2.imshow('Pre-Processed (Contrast Inc)', frame_eq)
        cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        cv2.imshow('Thresholded', adapt_thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    stop_car()
    picam2.stop()
    cv2.destroyAllWindows()
