#pragma once

#include "../servo/PCA9685.h"
#include "../util/util.h"
#include <vector>

namespace control {

    /*
        Contains defaults which fit to most servos
    */
    struct Servo {
        int servoNum = -1;
        double minAngle = -90;
        double maxAngle = 90;
        double minMs = 1.0;
        double maxMs = 2.0;
        double resetAngle = 0.0;
    };

    class ServoKit
    {
        
    private:

        struct Servo servos[16];
        uint32_t _reference_clock_speed = 25000000;
        int _frequency = 50;
        double _res = 4096;
        PCA9685 *_pwm;
        
        int millisecondToTicks(double impulseMs);
        int secondsToTicks(double impulseSec);

    public:

        /***
         * @brief Constructor
         * @param pwm - send in i2c PCA9685 object
         */
        ServoKit(PCA9685 *pwm);
        
        /***
         * @brief Sends set the angle of servo
         * @param servo - servo number
         * @param angle - angle 
         * @return A boolean indicating if operation was success 
         */
        bool setAngle(int servo, double angle);

        /***
         * @brief Set milisecond range of servo per servo spec
         * @param servo - servo number
         * @param low - min Ms
         * @param high - max Ms
         */
        void setMsRange(int servo, double low, double high);
        
        /***
         * @brief Set angle '+' and '-' range of servo per servo spec
         * @param servo - servo number
         * @param low - min angle (usually negative)
         * @param high - max angle (usually positive)
         */
        void setAngleRange(int servo, double low, double high);

        /***
         * @brief Initialize servo on PCA9685
         * @param servo - servo to init
         */
        void initServo(struct Servo servo);
    };
};
