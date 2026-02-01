import argparse
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


def drive_fwd(ds):
    print('Moving forwards')
    set_duty_cycle_both(ds)
    left_dir.forward()
    right_dir.forward()


def drive_bckwd(ds):
    print('Moving backwards')
    set_duty_cycle_both(ds)
    left_dir.backward()
    right_dir.backward()


def speed2dutycycle(time, speed):
    if time <= 1.5:
        duty_cycle = (speed +
            1.16786) / 0.720071  # from linear regression of testing data
        return (duty_cycle / 100)
    else:
        duty_cycle = (speed +
            12.24778) / 0.95181  # from linear regression of testing data
        return (duty_cycle / 100)


# Setup command-line arguement parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    '-t',
    '--time',
    type=float,
    required=True,
    help='Amount of time in seconds the car is to move')
parser.add_argument(
    '-d',
    '--direction',
    type=str,
    required=True,
    help='The direction the car is to move. Arguement passed must be either the letter "F" or "B", or the word "Forward" or "Backward"'
)
mvmnt = parser.add_mutually_exclusive_group()
mvmnt.add_argument(
    '-d',
    '--duty-cycle',
    type=float,
    help='Duty cycle to drive the car at, as a percentage')
mvmnt.add_argument(
    '-s', '--speed', type=float, help='Speed to drive the car at, in cm/s')
args = parser.parse_args()
# print(args)
# print(args.duty_cycle or args.speed)

if args.time < 0:
    raise argparse.ArgumentTypeError(
        'Time passed to program cannot be a negative value.')

if (args.duty_cycle or args.speed) is None:
    raise argparse.ArgumentTypeError(
        'Either speed or duty cycle must be specified')

duty_cycle = 0

if args.speed is None:  # input is duty cycle
    print('Duty cycle has been specified\n')
    duty_cycle = args.duty_cycle / 100
else:
    print('Speed has been specified\n')
    if args.speed > 71:
        raise argparse.ArgumentTypeError('Maximum speed is 71 cm/s')
    duty_cycle = speed2dutycycle(args.time, args.speed)

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

print(
    f'Input time      (seconds) : {args.time}\nInput duty cycle      (%) : {args.duty_cycle}\nInput speed        (cm/s) : {args.speed}\nCalculated duty cycle (%) : {duty_cycle * 100:.3f}\n'
)

set_duty_cycle_both(duty_cycle)
left_dir.forward()
right_dir.forward()

if len(args.direction) > 1:
    match set(['FORWARD',
        'BACKWARD']).intersection(set([args.direction.upper()])).pop():
        case 'FORWARD':
            drive_fwd(duty_cycle)
        case 'BACKWARD':
            drive_bckwd(duty_cycle)
        case _:
            raise argparse.ArgumentTypeError(
                f'"--direction" arguement [{args.direction}] is invalid')
else:
    match args.direction.upper():
        case 'F':
            drive_fwd(duty_cycle)
        case 'B':
            drive_bckwd(duty_cycle)
        case _:
            raise argparse.ArgumentTypeError(
                f'"--direction" arguement [{args.direction}] is invalid')

sleep(args.time)
