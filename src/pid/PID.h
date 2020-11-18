#pragma once
#include <chrono>
#include <iostream>
#include <cmath>
#include "../util/util.h"


struct PIDState {
	double e; // error
	double i; // Integral of error with respect to time
	double d; // Derivative of error with respect to time
	double din; // Delta input
	double dt; // Delta time
	double de; // Delta error

	void getStateArray(double state[2]) {
		state[0] = i;
		state[1] = din;

		// state[0] = i;
		// state[1] = d;
		// state[2] = e;
		// state[3] = din;
		// state[4] = dt;
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
		state mockUpdate(double input, double sleep = 0.0, bool normalize = true); // Non-binding "what if" update call, returning copies of internal pid variables
		
		// Normalize: Scale from ~ -1 to ~ 1 
		// Get current internal variable state. Call after update().
		state getState(bool normalize = true); 

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

		double _prevError;
		double _integral;
		double _deltaInput;
		double _deltaTime;
		double _deltaError;

		double _windup_guard;
		double _setpoint;
		double _last_input;

};

