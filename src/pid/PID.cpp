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

}

void PID::init() {

    _currTime = std::chrono::steady_clock::now();
    _prevTime = _currTime;

    // initialize the previous error
    _prevError1 = 0.0;
    _prevError2 = 0.0;
    _last_input = 0.0;
    _sumError = 0.0;
    _d2Error = 0.0;

    // initialize the term result variables
    _cP = 0.0;
    _cI = 0.0;
    _cD = 0.0;

    // Reset the gains 
    _kP = _init_kP;
    _kI = _init_kI;
    _kD = _init_kD;

}

double PID::update(double input, double sleep) {
    
    // Delta time
    _currTime = std::chrono::steady_clock::now();
    _deltTime = _currTime - _prevTime;

    _deltaTime = double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

    // Delay execution to rate
    if (sleep > _deltaTime * 1000.0) {
        double diff = sleep - _deltaTime;
        _prevTime = _currTime;
        Utility::msleep(static_cast<long>(diff));
        _currTime = std::chrono::steady_clock::now();
        _deltTime = _currTime - _prevTime;
        _deltaTime = double(_deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
    }

    // Error
    double error = input - _setpoint;

    // Proportional of Error
    _cP = error;

    // Integral of error with respect to time
    _sumError += error;
    _cI += (error * (_kI * _deltaTime));

    // Derivative of input with respect to time
    _deltaError = error - _prevError1;
    _deltaInput = (_last_input - input);
    (_deltaTime > 0.0) ? (_cD = (1.0 / _deltaTime)) : (_cD = 0.0);

    // Integral windup gaurd
    if (_cI < -_windup_guard) {
        _cI = -_windup_guard;
    }
    else if (_cI > _windup_guard) {
        _cI = _windup_guard;
    }

    // save previous time and error
    _d2Error = error - ( 2.0 * _prevError1 ) + _prevError2;
    _prevTime = _currTime;
    _last_input = input;
    _prevError2 = _prevError1;
    _prevError1 = error; 

    // Cross-mult, sum and return
    double output = (_kP * _cP) + (_cI) - (_kD * _cD * _deltaInput);

    // Enforce PID gain bounds
    if (output > _max) {
        output = _max;
    }
    else if (output < _min) {
        output = _min;
    }

    return output;
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

std::vector<double> PID::getState() {
    double errorBound = 2.0 * _setpoint;
    std::vector<double> stateData;
    
    stateData.push_back(Utility::normalize(_cP, -errorBound, errorBound));
    stateData.push_back(Utility::normalize(_cI, -_windup_guard, _windup_guard));
    stateData.push_back(Utility::normalize(-(_cD * _deltaInput), -2.0 * errorBound, 2.0 * errorBound));
    stateData.push_back(Utility::normalize(_deltaInput, -2.0 * errorBound, 2.0 * errorBound));
    stateData.push_back(_deltaTime);
    stateData.push_back(Utility::normalize(_deltaError, -2.0 * errorBound, 2.0 * errorBound));
    stateData.push_back(std::clamp<double>(Utility::normalize(_sumError, -errorBound, errorBound), -1.0, 1.0));

    return stateData;
}

state PID::getState(bool normalize)
{
    state g;

    // Delta time
    // g.dt = _deltaTime;

    // Error
    g.e = _cP;

    // Delta error
    g.de = _deltaError;

    // Integral of error with respect to time
    // g.i = _cI;

    // Sum of error
    g.errSum = _sumError;

    // Delta input
    // g.din = _deltaInput;

    // Negative Derivative of input with respect to time.
    // g.d = -(_cD * _deltaInput);

    g.d2e = _d2Error;

    // Normalize and scale datapoints. 
    if (normalize) {
        double errorBound = 2.0 * _setpoint;
        g.e = Utility::normalize(g.e, -errorBound, errorBound);
        g.de = Utility::normalize(g.de, -errorBound, errorBound);
        // g.din = Utility::normalize(g.din, -2.0 * errorBound, 2.0 * errorBound);
        // g.i = Utility::normalize(g.i, -_windup_guard, _windup_guard);
        // g.d = Utility::normalize(g.d, -2.0 * errorBound, 2.0 * errorBound);
        g.errSum = std::clamp<double>(Utility::normalize(g.errSum, -errorBound, errorBound), -1.0, 1.0);
        g.d2e = Utility::normalize(g.d2e, -errorBound, errorBound);

        // g.e = g.e / errorBound;
        // g.de = g.de / errorBound;
        // g.errSum = std::clamp<double>(g.errSum / errorBound, -1.0, 1.0);
        // g.d2e = g.d2e / errorBound;
    }

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
    _kP = 0;
    _kI = 0;
    _kD = 0; 
    _kP += kP;
    _kI += kI;
    _kD += kD; 
}


