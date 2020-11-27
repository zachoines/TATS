#include "PanTiltEnv.h"

namespace TATS {
    
    PanTiltEnv::PanTiltEnv() {

        // Setup I2C and PWM
        _wire = new control::Wire();
        _pwm = new control::PCA9685(0x40, _wire);
        _servos = new control::ServoKit(_pwm);
        _config = new Utility::Config();
        
        _currentSteps = 0;

        for (int servo = 0; servo < _numServos; servo++) {
            double setpoint = static_cast<double>(_config->dims[servo]) / 2.0;
            _pids[servo] = new PID(	_config->defaultGains[0], 
                                    _config->defaultGains[1], 
                                    _config->defaultGains[2], 
                                    _config->pidOutputLow, 
                                    _config->pidOutputHigh, 
                                   setpoint);

            _resetAngles[servo] = _config->resetAngles[servo];
            _disableServo[servo] = _config->disableServo[servo];
            _invert[servo] = _config->invertServo[servo];
            _currentAngles[servo] = _config->resetAngles[servo];
            _lastAngles[servo] = _config->resetAngles[servo];

            _servos->initServo({
                .servoNum = servo,
                .minAngle = -98.5, 
                .maxAngle = 98.5,
                .minMs = 0.553,
                .maxMs = 2.520,
                .resetAngle = _resetAngles[servo]
            });
        }
    }

    PanTiltEnv::~PanTiltEnv() {
        delete _wire;
        delete _pwm;
        delete _servos;
        delete _config;
    }

    void PanTiltEnv::_sleep()
    {
        int milli = 1000 / _config->updateRate;
        std::this_thread::sleep_for(std::chrono::milliseconds(milli));
    }

    void PanTiltEnv::_syncEnv() {
        _sleep();

        std::unique_lock<std::mutex> lck(_lock);

        // Sleep for specified time and Wait for env to respond to changes
        try {

            while (_currentTimeStamp == _lastTimeStamp) {
                _cond.wait(lck);
            }

            _lastData = _currentData;
            _currentData = _eventData;
            _lastTimeStamp = _currentTimeStamp;

            lck.unlock();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            throw std::runtime_error("cannot sync with servos");
        }
    }

    void PanTiltEnv::_resetEnv() {
        for (int servo = 0; servo < _numServos; servo++) {
            _servos->setAngle(servo, _resetAngles[servo]);
            _lastAngles[servo] = _resetAngles[servo];
            _currentAngles[servo] = _resetAngles[servo];
            _pids[servo]->init();
        }
    }

    bool PanTiltEnv::isDone() {
        return _currentData[Data::done];
    }

    StateData PanTiltEnv::reset() {
        _resetEnv();
        _syncEnv();

        StateData data;

        // Push back tilt  and pan data
        for (int servo = 0; servo < _numServos; servo++) {
            std::vector<double> pidState = _pids[servo]->getState();
            data.insert(data.end(), pidState.begin(), pidState.end());
            data.push_back(_currentData[servo + 2] / _currentData[servo + 4]);
            data.push_back(_lastAngles[servo] / _config->anglesHigh[servo]);
            data.push_back(_currentAngles[servo] / _config->anglesHigh[servo]);
        }
        
        return data;
    }

    Transition PanTiltEnv::step(Actions actions, bool rescale = true) {
        _currentSteps = (_currentSteps + 1) % INT_MAX; 
        Actions rescaledActions = actions;
        Transition stepResults;
        StateData nextState;
        
        for (int servo = 0; servo < _numServos; servo++) {

            if (_disableServo[servo]) {
                continue;
            }

            stepResults.actions[servo] = actions[servo];

            if (rescale) {
                rescaledActions[servo] = Utility::rescaleAction(actions[servo], _config->actionLow, _config->actionHigh);
            }

            double newAngle = rescaledActions[servo];
            
            if (_invert[servo]) { 
                newAngle = newAngle * -1.0; 
            }

            _lastAngles[servo] = _currentAngles[servo];
            _currentAngles[servo] = newAngle;
            _servos->setAngle(servo, newAngle);
        }

        _syncEnv();

        for (int servo = 0; servo < _numServos; servo++) {

            if (_disableServo[servo]) {
                continue;
            }		

            double lastError = _lastData[servo];
            double currentError = _currentData[servo];

            if (_lastData[Data::done]) {
                throw std::runtime_error("State must represent a complete transition");
            }
            else {
                stepResults.reward += Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_config->dims[servo]) / 2.0, _currentData[Data::done], 0.01, false);
            }

            // Update to get the next state
            _pids[servo]->update(currentError, 1000.0 / static_cast<double>(_config->updateRate));

            // Then push back new tilt data
            std::vector<double> pidState = _pids[servo]->getState();
            nextState.insert(nextState.end(), pidState.begin(), pidState.end());
            nextState.push_back(_currentData[servo + 2] / _currentData[servo + 4]);
            nextState.push_back(_lastAngles[servo] / _config->anglesHigh[servo]);
            nextState.push_back(_currentAngles[servo] / _config->anglesHigh[servo]);            
        }

        stepResults.nextState = nextState;
        stepResults.empty = false;
        stepResults.done = _currentData[Data::done];

        return stepResults;
    }

    void PanTiltEnv::update(EventData data, double timeStamp) {
        std::unique_lock<std::mutex> lck(_lock);
        try
        {
            _eventData = data;
            _lastTimeStamp = timeStamp;
            
            lck.unlock();
            _cond.notify_all();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            throw std::runtime_error("cannot update event data");
        }
    }
    
};
