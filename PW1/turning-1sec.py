import argparse
# from sys import exit, stderr
from gpiozero import Motor, PWMOutputDevice
from time import sleep
from typing import Union


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


def angle2dutycycle(angle: Union[int, float]) -> Union[int, float]:
    out = (angle + 94.28571) / 224.4898
    return out


def turn_right(duty_cycle: Union[int, float]) -> None:
    print("Turning Right")
    set_duty_cycle_left(duty_cycle)
    set_duty_cycle_right(0)
    left_dir.forward()
    right_dir.stop()


def turn_left(duty_cycle: Union[int, float]) -> None:
    print("Turning Left")
    set_duty_cycle_left(0)
    set_duty_cycle_right(duty_cycle)
    left_dir.stop()
    right_dir.forward()


# Setup command-line arguement parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--angle",
    type=int,
    required=True,
    help=
    "Amount in degrees to turn the car left. Argument passed must be a positive number."
)
parser.add_argument(
    "-d",
    "--direction",
    type=str,
    required=True,
    help=
    "The direction the car is to turn. Arguement passed must be either the letter \"L\" or \"R\", or the word \"Left\" or \"Right\""
)
args = parser.parse_args()
# print(args)

if args.angle > 135:
    raise argparse.ArgumentTypeError(
        "Turn angles exceeding 135 degrees are not supported.")
if args.angle < 0:
    raise argparse.ArgumentTypeError(
        "Negative turn angles are not supported. To change the direction of motion, please pass a \"direction\" arguement"
    )

ENA = 13  # Control right side motors; GPIO/BCM pin 13, Physical/Board pin 33
ENB = 19  # Control left side motors;  GPIO/BCM pin 19, Physical/Board pin 35

DUTY_CYCLE = 0
match args.angle:
    case 90:
        DUTY_CYCLE = 0.85
    case 45:
        DUTY_CYCLE = 0.60
    case _:
        DUTY_CYCLE = angle2dutycycle(args.angle)
        # raise Exception(f"A turn angle of {args.angle} is not supported")
        # print(f"ERROR: A turn angle of {args.angle} is not supported",
        #       file=stderr)
        # exit(1)

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

if len(args.direction) > 1:
    match set(["LEFT",
               "RIGHT"]).intersection(set([args.direction.upper()])).pop():
        case "LEFT":
            turn_left(DUTY_CYCLE)
        case "RIGHT":
            turn_right(DUTY_CYCLE)
        case _:
            raise argparse.ArgumentTypeError(
                f"\"--direction\" arguement [{args.direction}] is invalid")
else:
    match args.direction.upper():
        case "L":
            turn_left(DUTY_CYCLE)
        case "R":
            turn_right(DUTY_CYCLE)
        case _:
            raise argparse.ArgumentTypeError(
                f"\"--direction\" arguement [{args.direction}] is invalid")

sleep(1)
