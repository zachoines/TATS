
#pragma once

#include "EnvBase.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "../util/data.h"
#include "../wire/Wire.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"

/*
    This is a base class only
*/

namespace TATS {

    struct PIDConfig {
        // PID options
        double pidOutputHigh;
        double pidOutputLow;
        double defaultGains[3];
        double setpoint;

        PIDConfig() :
            pidOutputHigh(60.0),                 // Max output allowed for PID's
            pidOutputLow(-60.0),				 // Min output allowed for PID's
            defaultGains({ 0.05, 0.04, 0.001 }), // Gains fed to pids when initialized    
            setpoint(0.0)

            {}

    } typedef pidcfg;

    struct ServoConfig {

        // Servo options
        bool invertServo;
        bool disableServo;
        double resetAngle;
        double angleHigh;
        double angleLow;

        // Servo specs
        double minAngle;
        double maxAngle;
        double minMs;
        double maxMs;
    
        ServoConfig() :

            disableServo(false),                 // Disable the servo
            invertServo(false),                  // Flip output angle of servo
            resetAngle(0.0),                     // Angle when reset
            angleHigh(60.0),                     // Max allowable output angle to servos
            angleLow(-60.0),                     // Min allowable output angle to servos
                   
            minAngle(-98.5), 
            maxAngle(98.5),
            minMs(0.553),
            maxMs(2.520)
            
            {}

    } typedef scfg;

    // Raw update data 
    enum Data {
        error = 1,   
        obj = 2,
        frame = 3,        
        done = 4
    };

    // The formal state 
    enum State {
        e = 0,        // error
        i = 1,        // Integral of error with respect to time
        d = 2,        // negative derivative of input with respect to time
        din = 3,      // Delta input
        dt = 4,       // Delta time
        de = 5,       // Delta error
        errSum = 6,   // Sum of error
        obj = 7,      // Object location
        la = 8,       // last angle
        ca = 9        // current angle
    };

    class ServoEnv : EnvBase
    {
    private:

        static const int _stateSize = 10;
        int _servoNum;
        TATS::scfg* _servo_config;
        TATS::pidcfg* _pid_config;
        Utility::cfg* _global_config;

        control::ServoKit* _servos;
        PID* _pid;

        std::mutex _lock;
        std::condition_variable _cond;

        double _lastTimeStamp;
        double _currentTimeStamp;

        double _currentAngle;
        double _lastAngle;
        int _currentSteps; 

        EventData _eventData;
        EventData _lastData;
        EventData _currentData;

        void _sleep(int rate);
        void _syncEnv();
        void _resetEnv();

    public:
        ServoEnv(control::ServoKit* servokit, int servo_num);
        ~ServoEnv();

        bool isDone();
        StateData reset(); 
        Transition step(Actions actions, bool rescale = true);
        void update(StateData data, double timeStamp); 
    };
};
