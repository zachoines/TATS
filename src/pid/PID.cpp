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

	_windup_guard = 1000; 
	_setpoint = setpoint;

}

void PID::init() {

	_currTime = std::chrono::steady_clock::now();
	_prevTime = _currTime;

	// initialize the previous error
	_prevError = 0.0;
	_last_input = 0.0;
	_integral = 0.0;

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
	_integral += error * _deltaTime;
	_cI += (error * (_kI * _deltaTime));

	// Derivative of input with respect to time
	_deltaError = error - _prevError;
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
	_prevTime = _currTime;
	_last_input = input;
	_prevError = error;

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

// A nonbinding, 'What if', update call to PID. Returns internal PID state variables.
state PID::mockUpdate(double input, double sleep, bool normalize)
{
	std::chrono::steady_clock::time_point currTime = std::chrono::steady_clock::now();
	std::chrono::steady_clock::duration deltTime = currTime - _prevTime;
	double dTime = double(deltTime.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
	state g;
	
	// Delta time
	g.dt = dTime;

	// Error
	g.e = input - _setpoint;

	// Delta input
	g.din = (_last_input - input);

	// Integral of error with respect to time
	g.i = _integral + (g.e * dTime);

	// Negative Derivative of input with respect to time.
	g.d = (dTime > 0.0) ? -1.0 * (g.din / dTime) : 0.0;

	// Just divide by the max possible values, preserve sign
	if (normalize) {
		g.e /= _setpoint;
		g.din /= 2.0 * _setpoint;
		g.i /= _windup_guard;
		g.d /= _windup_guard;
	}
	
	return g;
}

state PID::getState(bool normalize)
{
	state g;

	// Delta time
	g.dt = _deltaTime;

	// Error
	g.e = _cP;

	// Integral of error with respect to time
	g.i = _cI;

	// Delta input
	g.din = _deltaInput;

	// Negative Derivative of input with respect to time.
	g.d = (_deltaTime > 0.0) ? -1.0 * (_cD * _deltaInput) : 0.0;

	// Just divide by the max possible values, preserve sign
	if (normalize) {
		g.e /= _setpoint;
		g.din /= 2.0 * _setpoint;
		g.i /= _windup_guard;
		g.d /= _windup_guard;
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
	_kP = kP;
	_kI = kI;
	_kD = kD; 
}


