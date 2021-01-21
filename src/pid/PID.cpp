#include "PID.h"
#include "../util/util.h"

PID::PID(double kP, double kI, double kD, double min, double max, double setpoint) {
    _kP = kP;
    _kI = kI;
    _kD = kD;
 
    _init_kP = kP;
    _init_kI = kI;
    _init_kD = kD;

    _max = max;
    _min = min;

    _windup_guard = setpoint * 2; 
    _setpoint = setpoint;

    for (int i = 0; i < 4; i++) {
        _prevErrors[i] = 0.0;
        _prevOutputs[i] = 0.0;
    }
}

void PID::init() {

    _currTime = std::chrono::steady_clock::now();
    _prevTime = _currTime;

    // initialize the previous error
    _prevError = 0.0;
    _last_input = 0.0;
    _sumError = 0.0;

    // initialize the term result variables
    _cP = 0.0;
    _cI = 0.0;
    _cD = 0.0;

    // Reset the gains 
    _kP = _init_kP;
    _kI = _init_kI;
    _kD = _init_kD;

    // Reset deltas
    _deltaError = 0.0;
    _deltaInput = 0.0;
    _deltaTime = 0.0;

    for (int i = 0; i < 4; i++) {
        _prevErrors[i] = 0.0;
        _prevOutputs[i] = 0.0;
    }

}

double PID::update(double input) {
    
    // Delta time
    _currTime = std::chrono::steady_clock::now();
    _deltTime = _currTime - _prevTime;
    _deltaTime = std::clamp<double>(double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den, 0.0, 1.0);

    // Error
    double error = input - _setpoint;

    // Proportional of Error
    _cP = error;

    // Integral of error with respect to time
    _sumError += (error * _deltaTime);
    _cI += (error * (_kI * _deltaTime));

    // Derivative of input with respect to time
    _deltaInput = (_last_input - input);
    (_deltaTime > 0.0) ? (_cD = (1.0 / _deltaTime)) : (_cD = 0.0);

    // Integral windup gaurd
    _cI = std::clamp<double>(_cI, -_windup_guard, _windup_guard);

    // save previous time, error, and outputs
    _prevTime = _currTime;
    _last_input = input;
    _prevError = error; 

    for (int i = 2; i >= 0; i--) {
        _prevOutputs[i + 1] = _prevOutputs[i];
        _prevErrors[i + 1] = _prevErrors[i];
    }

    // Cross-mult, sum and return, enforce PID gain bounds
    _prevOutputs[0] = std::clamp<double>((_kP * _cP) + (_cI) - (_kD * _cD * _deltaInput), _min, _max);
    _prevErrors[0] = error;

    return _prevOutputs[0];
}

void PID::getPID(double w[3])
{
    w[0] = _cP;
    w[1] = _cI;
    w[2] = _cD;
}

void PID::setWindupGaurd(double guard)
{
    _windup_guard = guard;
}

double PID::getWindupGaurd()
{
    return _windup_guard;
}

state PID::getState(bool normalize)
{
    state g = {};

    g.i = 0;
    for (int i = 0; i < 4; i++) {
        g.errors[i] = _prevErrors[i];
        g.outputs[i] = _prevOutputs[i];
        g.i += (g.errors[i] * _deltaTime);
    }

    g.din = _deltaInput;
    g.in = _last_input;
    g.dt = _deltaTime;
    g.setPoint = _setpoint;

    return g;
}

void PID::getWeights(double w[3])
{
    w[0] = _kP;
    w[1] = _kI;
    w[2] = _kD;
}

void PID::setWeights(double kP, double kI, double kD)
{
    _kP = kP;
    _kI = kI;
    _kD = kD; 
}


