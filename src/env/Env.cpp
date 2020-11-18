#include "Env.h"

namespace TATS {
	Env::Env()
	{
		// Setup I2C and PWM
		_wire = new control::Wire();
		_pwm = new control::PCA9685(0x40, _wire);
		_servos = new control::ServoKit(_pwm);
		_config = new Utility::Config();
		_currentSteps = 0;

		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			_pids[servo] = new PID(	_config->defaultGains[0], 
									_config->defaultGains[1], 
									_config->defaultGains[2], 
									_config->pidOutputLow, 
									_config->pidOutputHigh, 
									static_cast<double>(_config->dims[servo]) / 2.0);
			
			_resetAngles[servo] = _config->resetAngles[servo];
			_disableServo[servo] = _config->disableServo[servo];
			_invert[servo] = _config->invertServo[servo];
			_currentAngles[servo] = _config->resetAngles[servo];
			_lastAngles[servo] = _config->resetAngles[servo];

			_servos->initServo({
				.servoNum = servo,
				.minAngle = -72.0, 
				.maxAngle = 72.0,
				.minMs = 0.750,
				.maxMs = 2.250,
				.resetAngle = _resetAngles[servo]
			});
		}
	}

	Env::~Env() {
		delete _wire;
		delete _pwm;
		delete _servos;
		delete _config;
	}

	void Env::_sleep()
	{
		int milli = 1000 / _config->updateRate;
		std::this_thread::sleep_for(std::chrono::milliseconds(milli));
	}

	void Env::_syncEnv()
	{
		_sleep();

		std::unique_lock<std::mutex> lck(_lock);

		// Sleep for specified time and Wait for env to respond to changes
		try {
			for (int servo = 0; servo < NUM_SERVOS; servo++) {
				
				if (_disableServo[servo]) {
					continue;
				}				

				while (_eventData[servo].timestamp == _lastTimeStamp[servo]) {
					_cond.wait(lck);
				}

				_lastData[servo] = _currentData[servo];
				_currentData[servo] = _eventData[servo];
				_lastTimeStamp[servo] = _currentData[servo].timestamp;
			}

			lck.unlock();
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			throw std::runtime_error("cannot sync with servos");
		}
	}

	void Env::update(Utility::ED eventDataArray[NUM_SERVOS]) {
		
		std::unique_lock<std::mutex> lck(_lock);
		try
		{
			for (int servo = 0; servo < NUM_SERVOS; servo++) {
				if (_disableServo[servo]) {
					continue;
				}

				this->_eventData[servo] = eventDataArray[servo];
			}
			
			lck.unlock();
			_cond.notify_all();
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			throw std::runtime_error("cannot update event data");
		}
	}

	bool Env::isDone()
	{
		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			if (_disableServo[servo]) {
				continue;
			}
			else {
				return _currentData[servo].done;
			}
		} 

		return true;
	}

	void Env::_resetEnv()
	{

		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			_servos->setAngle(servo, _resetAngles[servo]);
			_lastAngles[servo] = _resetAngles[servo];
			_currentAngles[servo] = _resetAngles[servo];
			_pids[servo]->init();
		}
	}

	Utility::RD Env::reset()
	{
		_resetEnv();
		_syncEnv();

		Utility::RD data;

		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			_observation[servo].pidStateData = _pids[servo]->getState(false);
			// _observation[servo].obj = (_currentData[servo].frame > 0) ? _currentData[servo].obj / _currentData[servo].frame : 0.0;
			// _observation[servo].lastAngle = _lastAngles[servo] / _config->anglesHigh[servo];
			// _observation[servo].currentAngle = _currentAngles[servo] / _config->anglesHigh[servo];
			_observation[servo].obj = _currentData[servo].obj;
			_observation[servo].frame = _currentData[servo].frame;
			_observation[servo].lastAngle = _lastAngles[servo];
			_observation[servo].currentAngle = _currentAngles[servo];
			data.servos[servo] = _observation[servo];
		}

		return data;
	}

	// Using action, take step and return observation, reward, done, and actions for every servo. 
	// Note: SR[servo].currentState is always null. Retrieve currentState from previous 'step' or 'reset' call.
	Utility::SR Env::step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale)
	{
		_currentSteps = (_currentSteps + 1) % INT_MAX; 
		
		Utility::SR stepResults;

		double randChance = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
		
		for (int servo = 0; servo < NUM_SERVOS; servo++) {

			if (_disableServo[servo]) {
				
				// Used when tuning PIDs
				continue;
			}

			// Scale PID actions if configured
			for (int a = 0; a < NUM_ACTIONS; a++) {
				stepResults.servos[servo].actions[a] = actions[servo][a];

				if (rescale) {
					actions[servo][a] = Utility::rescaleAction(actions[servo][a], _config->actionLow, _config->actionHigh);
				}
			}

			// Print out the PID gains
			if (0.005 >= randChance) {
				std::cout << "Here are the new actions(s): ";
				for (int a = 0; a < _config->numActions; a++) {
					std::cout << actions[servo][a] << ", ";
				}
				std::cout << std::endl;
			}

			double newAngle = 0.0;
			if (!_config->usePIDs) {
				newAngle = actions[servo][0];
			}
			else {
				_pids[servo]->setWeights(actions[servo][0], actions[servo][1], actions[servo][2]);
				newAngle = _pids[servo]->update(_currentData[servo].obj, 1000.0 / static_cast<double>(_config->updateRate));
			}

			// newAngle = Utility::mapOutput(newAngle, _config->pidOutputLow, _config->pidOutputHigh, _config->angleLow, _config->angleHigh);
			if (_invert[servo]) { 
				newAngle = newAngle * -1.0; 
			}

			_lastAngles[servo] = _currentAngles[servo];
			_currentAngles[servo] = newAngle;

			// Print out the angles
			if (0.005 >= randChance) {
				std::cout << "Here are the angles: ";
				std::cout << newAngle << std::endl;
				std::cout << "For servo: ";
				std::cout << servo << std::endl;
			}

			_servos->setAngle(servo, newAngle);
		}

		_syncEnv();

		for (int servo = 0; servo < NUM_SERVOS; servo++) {

			if (_disableServo[servo]) {
				continue;
			}		

			double lastError = _lastData[servo].obj;
			double currentError = _currentData[servo].obj;

			if (_lastData[servo].done) {
				throw std::runtime_error("State must represent a complete transition");
			}
			else {
				stepResults.servos[servo].reward = Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_config->dims[servo]) / 2.0, _currentData[servo].done, 0.01, false);
			}

			// Fill out the step results
			if (!_config->usePIDs) {
				_pids[servo]->update(currentError, 1000.0 / static_cast<double>(_config->updateRate));
				stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(false);
			}
			else {
				stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(false);
			}

			// stepResults.servos[servo].nextState.obj = _currentData[servo].obj / _currentData[servo].frame;
			// stepResults.servos[servo].nextState.lastAngle = _lastAngles[servo] / _config->anglesHigh[servo];
			// stepResults.servos[servo].nextState.currentAngle = _currentAngles[servo] / _config->anglesHigh[servo];
			stepResults.servos[servo].nextState.obj = _currentData[servo].obj;
			stepResults.servos[servo].nextState.frame = _currentData[servo].frame;
			stepResults.servos[servo].nextState.lastAngle = _lastAngles[servo];
			stepResults.servos[servo].nextState.currentAngle = _currentAngles[servo];
			stepResults.servos[servo].done = _currentData[servo].done;
			stepResults.servos[servo].empty = false;
			_observation[servo] = stepResults.servos[servo].nextState;
		}

		return stepResults;
	}

	void Env::getDisabled(bool servos[NUM_SERVOS]) {
		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			servos[servo] = _disableServo[servo];
		}
	}
	
	void Env::setDisabled(bool servos[NUM_SERVOS]) {
		for (int servo = 0; servo < NUM_SERVOS; servo++) {
			_disableServo[servo] = servos[servo];
		}
	}
};