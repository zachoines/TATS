
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

namespace TATS {

    // Raw update data 
    enum Data {
        tiltError = 0, 
        panError = 1,   
        objCenterY = 2,
        objCenterX = 3,
        Y = 4,
        X = 5,
        done = 6
    };

    // The formal state 
    enum State {

        // Tilt entries
        e_t = 0,        // error
        i_t = 1,        // Integral of error with respect to time
        d_t = 2,        // negative derivative of input with respect to time
        din_t = 3,      // Delta input
        dt_t = 4,       // Delta time
        de_t = 5,       // Delta error
        errSum_t = 6,   // Sum of error
        obj_t = 7,      // Object location
        la_t = 8,       // last angle
        ca_t = 9,       // current angle

        // Pan entries
        e_p = 10,       // error
        i_p = 11,       // Integral of error with respect to time
        d_p = 12,       // negative derivative of input with respect to time
        din_p = 13,     // Delta input
        dt_p = 14,      // Delta time
        de_p = 15,      // Delta error
        errSum_p = 16,  // Sum of error
        obj_p = 17,     // Object location
        la_p = 18,      // last angle
        ca_p = 19,      // current angle
    };

    // Outputs 
    enum Action {
        tilt = 0,
        pan = 1
    };

    class PanTiltEnv : EnvBase
    {
    private:

        static const int _numServos = 2;
        static const int _stateSize = 20;

        control::Wire* _wire;
        control::PCA9685* _pwm;
        control::ServoKit* _servos;
        Utility::cfg* _config;
        PID* _pids[2];

        std::mutex _lock;
        std::condition_variable _cond;

        int _frameSkip;
        double _lastTimeStamp;
        double _currentTimeStamp;

        bool _invert[2];
        bool _disableServo[2];
        double _resetAngles[2];
        double _currentAngles[2];
        double _lastAngles[2];
        int _currentSteps; 
        bool _forceEnable;

        EventData _eventData;
        EventData _lastData;
        EventData _currentData;

        void _syncEnv();
        void _resetEnv();
        void _sleep();

    public:
        PanTiltEnv();
        ~PanTiltEnv();

        bool isDone();
        StateData reset(); 
        Transition step(Actions actions, bool rescale = true);
        void update(StateData data, double timeStamp); 
    };
};
