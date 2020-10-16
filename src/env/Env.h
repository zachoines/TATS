#pragma once
#include <cmath>

#include "../util/data.h"
#include "../util/util.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"
#include "../servo/ServoKit.h"
#include "../wire/Wire.h"
#include "../pid/PID.h"

namespace TATS {
	class Env
	{
	private:
		pthread_mutex_t* _lock;
		pthread_cond_t* _cond;
		int _frameSkip;
		Utility::param* _params;
		Utility::cfg* _config;

		double _lastTimeStamp[NUM_SERVOS];

		Utility::ED _eventData[NUM_SERVOS];
		Utility::ED _lastData[NUM_SERVOS];
		Utility::ED _currentData[NUM_SERVOS];
		Utility::SD _observation[NUM_SERVOS];

		bool _invert[NUM_SERVOS];
		bool _disableServo[NUM_SERVOS];
		double _resetAngles[NUM_SERVOS];
		double _currentAngles[NUM_SERVOS];
		double _lastAngles[NUM_SERVOS];

		PID* _pan = nullptr;
		PID* _tilt = nullptr;
		PID* _pids[NUM_SERVOS];

		void _sleep();
		void _syncEnv();
		void _resetEnv(); // Resets servos and re-inits PID's. Call only once manually.

		control::Wire* _wire;
		control::PCA9685* _pwm;
		control::ServoKit* _servos;

	public:
		Env(Utility::param* parameters, pthread_mutex_t* stateDataLoc, pthread_cond_t* stateDataCond);
		~Env();

		bool isDone();

		Utility::RD reset();  // Steps with actions and waits for Env to update, then returns the current state.
		Utility::SR step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale = true);  // Steps with actions and waits for Env to update, then returns the current state.
		void update(Utility::ED eventDataArray[NUM_SERVOS]); // Called from another thread to update state variables
	};
};
