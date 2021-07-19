#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
// #include "../util/util.h"


struct PIDState {
    double dt; // Delta time
    double i; // Integral
    double din; // derivative
    double in; // input
    double setPoint;
    double errors[4]; // Previous errors
    double outputs[4]; // Previous outputs

    PIDState() : 
        dt(0.0),
        i(0.0),
        din(0.0),
        in(0.0),
        setPoint(0.0),
        errors({ 0.0 })
    {}

} typedef state;

class PID
{
    public:
        PID(double kP, double kI, double kD, double min, double max, double setpoint);
        void init();
        double update(double input, bool invert=false);
        void getWeights(double w[3]);
        void setWeights(double kP, double kI, double kD);
        void getPID(double w[3]);
        void setWindupGaurd(double guard);
        double getWindupGaurd();
        
        // Get current internal variable state. Call after update().
        state getState(bool normalize); 

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

        double _prevError;
        double _sumError;
        double _deltaInput;
        double _deltaTime;
        double _deltaError;
        double _prevErrors[4];
        double _prevOutputs[4];

        double _windup_guard;
        double _setpoint;
        double _last_input;

};

