#pragma once
#include <cmath>
#include <random>
#include <condition_variable>
#include <mutex>
#include <bits/stdc++.h>

#include "../util/data.h"
#include "../util/util.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"
#include "../wire/Wire.h"
#include "../pid/PID.h"

namespace TATS {
    class Env
    {
    private:

        std::mutex _lock;
        std::condition_variable _cond;
        Utility::cfg* _config;

        int _frameSkip;
        double _lastTimeStamp[NUM_SERVOS];

        Utility::ED _eventData[NUM_SERVOS] = {{}};
        Utility::ED _lastData[NUM_SERVOS] = {{}};
        Utility::ED _currentData[NUM_SERVOS] = {{}};
        Utility::SD _stateData[NUM_SERVOS] = {{}};

        bool _invert[NUM_SERVOS];
        bool _disableServo[NUM_SERVOS];
        double _resetAngles[NUM_SERVOS];
        double _currentAngles[NUM_SERVOS] = { 0.0 };
        double _lastAngles[NUM_SERVOS] = { 0.0 };
        double _errors[NUM_SERVOS][10] = { 0.0 };
        double _outputs[NUM_SERVOS][10] = { 0.0 };
        double _predObjLoc[NUM_SERVOS] = { 0.0 };
        int _currentSteps; 
        bool _forceEnable;

        PID* _pids[NUM_SERVOS];

        void _sleep(double rate=5);
        void _syncEnv(double rate=5);
        void _resetEnv(bool overrideResetAngles, double angles[NUM_SERVOS]); // Resets servos and re-inits PID's. Call only once manually.

        control::Wire* _wire;
        control::PCA9685* _pwm;
        control::ServoKit* _servos;

    public:
        Env();
        ~Env();

        bool isDone();
        void setDisabled(bool servos[NUM_SERVOS]);
        void getDisabled(bool servos[NUM_SERVOS]);
        Utility::RD reset(bool useCurrentAngles=false);  // Steps with actions and waits for Env to update, then returns the current state.
        Utility::RD reset(double angles[NUM_SERVOS]);  // Override reset with provided angles
        Utility::SR step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale = true, double rate = 5);  // Steps with actions and waits for Env to update, then returns the current state.
        void update(Utility::ED eventDataArray[NUM_SERVOS]); // Called from another thread to update state variables
        void getCurrentState(Utility::SD state[NUM_SERVOS]);
        void getPredictedObjectLocation(double locations[NUM_SERVOS]);
        void getCurrentAngle(double angles[NUM_SERVOS]);
    };
};
