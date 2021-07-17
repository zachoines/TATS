import Jetson.GPIO as GPIO
import time

led = 22

GPIO.setmode(GPIO.BOARD)  
GPIO.setup(led, GPIO.OUT, initial=GPIO.HIGH)

while True:
    GPIO.output(led, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(led, GPIO.HIGH)
    time.sleep(0.3)