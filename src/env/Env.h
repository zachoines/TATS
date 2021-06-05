#pragma once
#include <cmath>
#include <random>
#include <condition_variable>
#include <mutex>
#include <bits/stdc++.h>

#include "../util/config.h"
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

        bool _invertData[NUM_SERVOS];
        bool _invertAngles[NUM_SERVOS];
        bool _disableServo[NUM_SERVOS];
        double _resetAngles[NUM_SERVOS];
        double _currentAngles[NUM_SERVOS] = { 0.0 };
        double _lastAngles[NUM_SERVOS] = { 0.0 };
        double _errors[NUM_SERVOS][ERROR_LIST_SIZE] = { 0.0 };
        double _outputs[NUM_SERVOS][ERROR_LIST_SIZE] = { 0.0 };
        double _predObjLoc[NUM_SERVOS] = { 0.0 };
        int _currentSteps; 
        bool _recentReset;
        int _preSteps;

        PID* _pids[NUM_SERVOS];

        /***
         * @brief Sleeps this thread, 1 / rate 
         * @param rate - fraction of second
         * @return nothing
         */
        void _sleep(double rate=10);

        /***
         * @brief Sleeps for 1 / rate, then waits for signal that new data is available
         * @param rate - fraction of second
         * @return nothing
         */
        void _syncEnv(double rate=10);

        /***
         * @brief Set internal data and servos to defaults, with option to override angles
         * @param overrideResetAngles - override and default reset angles
         * @param angles - array with angels to reset env with 
         * @return nothing
         */
        void _resetEnv(bool overrideResetAngles, double angles[NUM_SERVOS]); // Resets servos and re-inits PID's. Call only once manually.

        control::Wire* _wire;
        control::PCA9685* _pwm;
        control::ServoKit* _servos;

    public:
        Env();
        ~Env();

        /***
         * @brief Returns true if the target is lost
         * @return a boolean
         */
        bool isDone();
        void setDisabled(bool servos[NUM_SERVOS]);
        void getDisabled(bool servos[NUM_SERVOS]);
        Utility::RD reset(bool useCurrentAngles=false);  // Steps with actions and waits for Env to update, then returns the current state.
        Utility::RD reset(double angles[NUM_SERVOS]);  // Override reset with provided angles


        /***
         * @brief Steps with actions and waits for Env to update, then returns the current state.
         * @param actions - 2D array containing the desired action foreach servo
         * @param rescale - rescale actions to range defined in config.h
         * @param rate - sleeps for 1 / rate
         * @return current state of env
         */
        Utility::SR step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale = true, double rate = 5);  
        void update(Utility::ED eventDataArray[NUM_SERVOS]); // Called from another thread to update state variables
        void getCurrentState(Utility::SD state[NUM_SERVOS]);
        void getPredictedObjectLocation(double locations[NUM_SERVOS]);
        void getCurrentAngle(double angles[NUM_SERVOS]);
    };
};
