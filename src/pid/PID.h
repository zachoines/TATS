#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include "../util/util.h"


struct PIDState {
    double e; // error
    double i; // Integral of error with respect to time
    double d; // negative derivative of input with respect to time
    double din; // Delta input
    double dt; // Delta time
    double de; // Delta error
    double d2e; // Second order error
    double errSum; // Sum of error

    void getStateArray(double state[5]) {
        state[0] = e;
        state[1] = de;
        state[2] = d2e; 
        state[3] = errSum; 
        state[4] = dt; 
    }

} typedef state;

class PID
{
    public:
        PID(double kP, double kI, double kD, double min, double max, double setpoint);
        void init();
        double update(double input, double sleep = 0.0);
        void getWeights(double w[3]);
        void setWeights(double kP, double kI, double kD);
        void getPID(double w[3]);
        void setWindupGaurd(double guard);
        double getWindupGaurd();
        
        // Get current internal variable state. Call after update().
        state getState(bool normalize); 
        std::vector<double> getState(); // Default normalized

    private:
        double _max;
        double _min;

        double _kP;
        double _kD;
        double _kI;

        double _cP;
        double _cI;
        double _cD;

        double _init_kP;
        double _init_kI;
        double _init_kD;

        std::chrono::steady_clock::time_point _currTime;
        std::chrono::steady_clock::time_point _prevTime;
        std::chrono::steady_clock::duration _deltTime;

        double _prevError1;
        double _prevError2;
        double _sumError;
        double _deltaInput;
        double _deltaTime;
        double _deltaError;
        double _d2Error;

        double _windup_guard;
        double _setpoint;
        double _last_input;

};

