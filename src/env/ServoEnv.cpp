#include "ServoEnv.h"

namespace TATS {
    
    ServoEnv::ServoEnv(control::ServoKit* servokit, int servo_num) {
        _servoNum = servo_num;
        _servos = servokit;
        _servo_config = new ServoConfig();
        _pid_config = new PIDConfig();
        _global_config = new Utility::Config();
        
        _pid = new PID(	_pid_config->defaultGains[0], 
                        _pid_config->defaultGains[1], 
                        _pid_config->defaultGains[2], 
                        _pid_config->pidOutputLow, 
                        _pid_config->pidOutputHigh, 
                        _pid_config->setpoint );
        

        _servos->initServo({
            .servoNum = servo_num,
            .minAngle = _servo_config->minAngle, 
            .maxAngle = _servo_config->maxAngle,
            .minMs = _servo_config->minMs,
            .maxMs = _servo_config->maxMs,
            .resetAngle = _servo_config->resetAngle
        });
        
    }

    ServoEnv::~ServoEnv() { 
        delete _pid;
        delete _global_config;
    };

    void ServoEnv::_sleep(int rate)
    {
        int milli = 1000 / rate;
        std::this_thread::sleep_for(std::chrono::milliseconds(milli));
    }

    void ServoEnv::_syncEnv() {
        _sleep(5);

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

    void ServoEnv::_resetEnv() {
        _servos->setAngle(_servoNum, _servo_config->resetAngle);
        _lastAngle = _servo_config->resetAngle;
        _currentAngle = _servo_config->resetAngle;
        _pid->init();
    }

    bool ServoEnv::isDone() {
        return _currentData[Data::done];
    }

    StateData ServoEnv::reset() {
        _resetEnv();
        _syncEnv();

        StateData data;
        std::vector<double> pidState = _pid->getState();
        data.insert(data.end(), pidState.begin(), pidState.end());
        data.push_back(_currentData[Data::obj] / _currentData[Data::frame]);
        data.push_back(_lastAngle / _servo_config->angleHigh);
        data.push_back(_currentAngle / _servo_config->angleHigh);
        
        return data;
    }

    Transition ServoEnv::step(Actions actions, bool rescale = true) {
        _currentSteps = (_currentSteps + 1) % INT_MAX; 
        Actions rescaledActions;
        Transition stepResults;
        StateData nextState;
        
        if (_servo_config->disableServo) {
            stepResults.empty = true;
            return stepResults;
        }

        stepResults.actions = actions;

        if (rescale) {
            rescaledActions.clear();
            for (double action : actions) {
                rescaledActions.push_back(Utility::rescaleAction(action, _global_config->actionLow, _global_config->actionHigh));
            }
        }

        double newAngle = 0.0;
        _pid->setWeights(rescaledActions[0], rescaledActions[1], rescaledActions[2]);
        newAngle = _pid->update(_currentData[Data::obj], 1000.0 / static_cast<double>(_global_config->updateRate));

        if (_servo_config->invertServo) { 
            newAngle = newAngle * -1.0; 
        }

        _lastAngle = _currentAngle;
        _currentAngle = newAngle;
        _servos->setAngle(_servoNum, newAngle);

        // Wait for the environment to respond to changes
        _syncEnv();

        double lastError = _lastData[Data::obj];
        double currentError = _currentData[Data::obj];

        if (_lastData[Data::done]) {
            throw std::runtime_error("State must represent a complete transition");
        }
        else {
            stepResults.reward = Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_pid_config->setpoint), _currentData[Data::done], 0.005, false);
        }

        // Then push back new tilt data
        std::vector<double> pidState = _pid->getState();
        nextState.insert(nextState.end(), pidState.begin(), pidState.end());
        nextState.push_back(_currentData[Data::obj] / _currentData[Data::frame]);
        nextState.push_back(_lastAngle / _servo_config->angleHigh);
        nextState.push_back(_currentAngle / _servo_config->angleHigh);            

        stepResults.nextState = nextState;
        stepResults.empty = false;
        stepResults.done = _currentData[Data::done];

        return stepResults;
    }

    void ServoEnv::update(EventData data, double timeStamp) {
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
