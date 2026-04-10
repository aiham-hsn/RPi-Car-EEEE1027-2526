from typing import Union
from numpy.typing import NDArray
from gpiozero import Motor, PWMOutputDevice
from picamera2 import Picamera2
import numpy as np
import libcamera
import cv2
import time
from math import copysign
from collections import deque, Counter


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


def stop_car():
    set_duty_cycle_both(0)
    left_dir.stop()
    right_dir.stop()


def process_frame(input_frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # Convert input frame to grayscale
    processed_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    processed_gray = cv2.GaussianBlur(processed_gray, (9, 9), 0)

    # Just use normal thresholding
    _, thresh = cv2.threshold(processed_gray, 120, 255, cv2.THRESH_BINARY_INV)

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
    contours, hierarchy = cv2.findContours(threshhold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def detect_colored_line(raw_frame, colour):
    hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
    if colour == "black":
        return False, None, None

    ylw_low, ylw_high = COLOUR_RANGES['yellow']
    ylw_mask = cv2.inRange(hsv, np.array(ylw_low), np.array(ylw_high))

    red_toplow, red_tophigh = COLOUR_RANGES['red']['top']
    red_btmlow, red_btmhigh = COLOUR_RANGES['red']['btm']
    red_masktop = cv2.inRange(hsv, np.array(red_toplow), np.array(red_tophigh))
    red_maskbtm = cv2.inRange(hsv, np.array(red_btmlow), np.array(red_btmhigh))
    red_mask = cv2.bitwise_or(red_maskbtm, red_masktop)

    match colour:
        case "yellow":
            accept_mask = process_frame_morphology_ops(ylw_mask)
            reject_mask = process_frame_morphology_ops(red_mask)
            colour_present = np.count_nonzero(accept_mask) > 0
            return bool(colour_present), accept_mask, reject_mask
        case "red":
            accept_mask = process_frame_morphology_ops(red_mask)
            reject_mask = process_frame_morphology_ops(ylw_mask)
            colour_present = np.count_nonzero(accept_mask) > 0
            return bool(colour_present), accept_mask, reject_mask
        case "both":
            both_mask = cv2.bitwise_or(ylw_mask, red_mask)
            both_mask = process_frame_morphology_ops(both_mask)
            colour_present = np.count_nonzero(both_mask) > 0
            return bool(colour_present), both_mask, None
        case _:
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


class ownPID:

    def __init__(self, Kp, Ki, Kd, setpoint=320):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0

    def update(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


pid = ownPID(0.5, 0.01, 0.05)
BASE_SPEED = 0.40
MIN_SPEED = 0.30

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

threshval = -1
global last_left_spd, last_right_spd, spd_diff
last_left_spd = last_right_spd = BASE_SPEED
spd_diff = 0

drive_fwd(0.4)
time.sleep(0.2)

start_time = last_time = last_arrow_detect_time = time.time()

global followed_colour, ini_colour_dir, colour_dir_check
followed_colour = False
colour_dir_check = False
ini_colour_dir = None

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

        height, width = np.shape(thresh)

        frame_roi = frame[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height * (1 - frame_discard_offset)):]
        #frame_roi_w_contours = frame_roi
        thresh_roi = thresh[int(height *
            (frame_discard_percentage - frame_discard_offset)):int(height * (1 - frame_discard_offset)):]

        colour_present, accept_mask, reject_mask = detect_colored_line(frame_roi, PRIORITY_COLOUR)

        if colour_present:
            followed_colour = True
            thresh_roi = accept_mask.copy()  # type: ignore
            col_main_cent_x, _ = calc_centroid(thresh2maincontour(thresh_roi))
            if colour_dir_check is False:
                ini_colour_dir = 'Right' if (col_main_cent_x > (cam_size_x / 2)) else 'Left'
                colour_dir_check = True
        if reject_mask is not None:
            thresh_roi = cv2.bitwise_xor(thresh_roi, reject_mask)

        if (now - last_time) > 2:
            last_time = time.time()
            print(
                f"Clr : [{colour_present}] || Flwd : [{followed_colour}] || Dir Chk : {colour_dir_check} || Dir : [{ini_colour_dir}]"
            )

        thresh_roi = process_frame_morphology_ops(thresh_roi)

        main_contour = thresh2maincontour(thresh_roi)

        if main_contour is not None:
            if colour_present is False and followed_colour is True:
                ls = last_left_spd
                rs = last_right_spd
                match ini_colour_dir:
                    case 'Right':
                        print('Exiting coloured line, moving [Right]')
                        ls = max((BASE_SPEED - 15), 0)
                        rs = max((BASE_SPEED + 15), 0)
                    case 'Left':
                        print('Exiting coloured line, moving [Left]')
                        ls = max((BASE_SPEED + 15), 0)
                        rs = max((BASE_SPEED - 15), 0)
                set_duty_cycle_left(ls)
                set_duty_cycle_right(rs)

                ini_colour_dir = None
                followed_colour = False
                colour_dir_check = False

                left_dir.forward()
                right_dir.forward()

                move_sec = 1
                print(f'Moving for {move_sec}sec')
                time.sleep(1)
            else:
                cent_x, cent_y = calc_centroid(main_contour)
                pid_out = pid.update(cent_x, dt)
                corr = (pid_out) / (100 * 2)
                #print(f"type(cent_x) : [{type(cent_x)}]")

                frame_roi_w_points = cv2.circle(frame_roi, (cent_x, int(cam_size_y / 2)), 4, (255, 0, 0), 4)
                #ls = max(min(BASE_SPEED - corr, MIN_SPEED), 0)
                #rs = max(min(BASE_SPEED + corr, MIN_SPEED), 0)
                ls = last_left_spd = max((BASE_SPEED - corr), 0)
                rs = last_right_spd = max((BASE_SPEED + corr), 0)
                set_duty_cycle_left(ls)
                set_duty_cycle_right(rs)
                left_dir.forward()
                right_dir.forward()
                #print(
                #    f"Line cent_x: [{cent_x}] | Corr: [{corr:.2f}] | LS: [{ls:.2f}] | RS: [{rs:.2f}] | LLS: [{last_left_spd:.2f}] | LRS: [{last_right_spd:.2f}]"
                #)
        else:
            print('\nLine no longer visible\n')
            stop_car()
            spd_diff = last_left_spd - last_right_spd  # +ve when supposed to take a right turn, -ve when supposed to take a left turn
            # print(f"Speed diff: [{spd_diff}]")
            turn_sign = copysign(1, spd_diff)
            match turn_sign:
                case 1:
                    set_duty_cycle_left(0)
                    set_duty_cycle_right(0.8)
                    # print("Going backwards, right")
                case -1:
                    set_duty_cycle_left(0.8)
                    set_duty_cycle_right(0)
                    # print("Going backwards, left")
                case _:
                    set_duty_cycle_both(0.6)
            left_dir.backward()
            right_dir.backward()
            time.sleep(0.2)
            stop_car()

        # Display the different frames
        # cv2.imshow('Original', frame)
        # cv2.imshow('Pre-Processed (Gray + Blur)', processed)
        # cv2.imshow('Thresholded', thresh)
        # cv2.imshow('Orignal ROI', frame_roi)
        cv2.imshow('Thresholded ROI', thresh_roi)
        # cv2.imshow(
        #     'ROI w/ contours',
        #     frame_roi_w_points  # pyright: ignore[reportPossiblyUnboundVariable]
        # )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected, stopping program...")
finally:
    stop_car()
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\n\nTIme Taken : [{time.time() - start_time}]")
