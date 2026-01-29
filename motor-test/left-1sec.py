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


# Setup arguement parsing
parser = argparse.ArgumentParser()
parser.add_argument("-t",
                    "--turn",
                    type=int,
                    required=True,
                    help="Amount in degrees to turn the car left")
args = parser.parse_args()
# print(args)

if args.turn > 135:
    raise argparse.ArgumentTypeError(
        "Turn angles exceeding 135 degrees are not supported")

ENA = 13  # Control right side motors; GPIO/BCM pin 13, Physical/Board pin 33
ENB = 19  # Control left side motors;  GPIO/BCM pin 19, Physical/Board pin 35

DUTY_CYCLE = 0
match args.turn:
    case 90:
        DUTY_CYCLE = 0.85
    case 45:
        DUTY_CYCLE = 0.60
    case _:
        DUTY_CYCLE = angle2dutycycle(args.turn)
        # raise Exception(f"A turn angle of {args.turn} is not supported")
        # print(f"ERROR: A turn angle of {args.turn} is not supported",
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

set_duty_cycle_left(0)
set_duty_cycle_right(DUTY_CYCLE)
left_dir.stop()
right_dir.forward()
sleep(1)
