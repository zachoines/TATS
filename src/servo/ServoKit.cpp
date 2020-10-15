#include "ServoKit.h"

namespace control {
    ServoKit::ServoKit(PCA9685 *pwm) {
        _pwm = pwm;
        _pwm->begin();
        _pwm->setOscillatorFrequency(_reference_clock_speed);
        _pwm->setPWMFreq(_frequency);  
    }

    int ServoKit::millisecondToTicks(double impulseMs)
    {
        double cycleMs = 1000.0 / static_cast<double>(_frequency);
        return static_cast<int>(4096 * impulseMs / cycleMs + 0.5);
    }

    int ServoKit::secondsToTicks(double impulseSec) {
        double pulselength = 1000000.0; 
        pulselength /= static_cast<double>(_frequency);
        pulselength /= 4096.0;  // 12 bits of resolution
        impulseSec *= static_cast<double>(1000000);  // seconds to us
        impulseSec /= pulselength;
        return static_cast<int>(impulseSec);
    }
    
    bool ServoKit::setAngle(int servo, double angle) {
        struct Servo s;
        
        if (servo < 0 || servo > 15) {
            throw std::runtime_error("Servo number out of range");
        } else {
            s = servos[servo];
        }
        
		double millis = Utility::mapOutput(angle, s.minAngle, s.maxAngle, s.minMs / 1000.0, s.maxMs / 1000.0);
        double test = Utility::mapOutput(angle, s.minAngle, s.maxAngle, s.minMs, s.maxMs);
		int ticks = secondsToTicks(millis);
        int testTicks = millisecondToTicks(test);
		_pwm->setPWM(s.servoNum, 0, ticks);
	}

    void ServoKit::setMsRange(int servo, double low, double high) {

    }

    void ServoKit::setAngleRange(int servo, double low, double high) {

    }

    void ServoKit::initServo(struct Servo servo) {
        if (servo.servoNum < 0 || servo.servoNum > 15) {
            throw std::runtime_error("servo number out of range");
        } else {
            servos[servo.servoNum] = servo;
        }
    }
};

