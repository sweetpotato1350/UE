import RPi.GPIO as GPIO
import time

# GPIO 핀 번호 설정
trig_pin = 17
echo_pin = 27
left_motor_pin1 = 18
left_motor_pin2 = 23
right_motor_pin1 = 24
right_motor_pin2 = 25

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setup(trig_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)
GPIO.setup(left_motor_pin1, GPIO.OUT)
GPIO.setup(left_motor_pin2, GPIO.OUT)
GPIO.setup(right_motor_pin1, GPIO.OUT)
GPIO.setup(right_motor_pin2, GPIO.OUT)


# 모터 제어 함수
def move_forward():
    GPIO.output(left_motor_pin1, GPIO.HIGH)
    GPIO.output(left_motor_pin2, GPIO.LOW)
    GPIO.output(right_motor_pin1, GPIO.HIGH)
    GPIO.output(right_motor_pin2, GPIO.LOW)


def move_backward():
    GPIO.output(left_motor_pin1, GPIO.LOW)
    GPIO.output(left_motor_pin2, GPIO.HIGH)
    GPIO.output(right_motor_pin1, GPIO.LOW)
    GPIO.output(right_motor_pin2, GPIO.HIGH)


def turn_left():
    GPIO.output(left_motor_pin1, GPIO.LOW)
    GPIO.output(left_motor_pin2, GPIO.HIGH)
    GPIO.output(right_motor_pin1, GPIO.HIGH)
    GPIO.output(right_motor_pin2, GPIO.LOW)


def turn_right():
    GPIO.output(left_motor_pin1, GPIO.HIGH)
    GPIO.output(left_motor_pin2, GPIO.LOW)
    GPIO.output(right_motor_pin1, GPIO.LOW)
    GPIO.output(right_motor_pin2, GPIO.HIGH)


def stop():
    GPIO.output(left_motor_pin1, GPIO.LOW)
    GPIO.output(left_motor_pin2, GPIO.LOW)
    GPIO.output(right_motor_pin1, GPIO.LOW)
    GPIO.output(right_motor_pin2, GPIO.LOW)


# 거리 측정 함수
def measure_distance():
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.1)

    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)

    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance


try:
    while True:
        dist = measure_distance()
        print("Distance:", dist, "cm")

        if dist < 20:  # 장애물이 감지되면
            stop()
            time.sleep(0.5)
            turn_right()  # 우회전하도록 변경 가능 (turn_left()로 변경하면 좌회전)
            time.sleep(1.0)
            move_forward()  # 장애물 피해서 이동
        else:
            move_forward()  # 장애물이 없으면 전진

except KeyboardInterrupt:
    GPIO.cleanup()