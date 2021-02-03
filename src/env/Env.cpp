#include "Env.h"

namespace TATS {
    Env::Env()
    {
        _eventData;
        _lastData;
        _currentData;

        // Setup I2C and PWM
        _wire = new control::Wire();
        _pwm = new control::PCA9685(0x40, _wire);
        _servos = new control::ServoKit(_pwm);
        _config = new Utility::Config();
        _currentSteps = 0;

        for (int servo = 0; servo < NUM_SERVOS; servo++) {
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

            _servos->initServo(_config->servoConfigurations[servo]);

            for (int i = 0; i < 5; i++) {
                _outputs[servo][i] = 0.0;
                _errors[servo][i] = 0.0;
            }
        }
    }

    Env::~Env() {
        delete _wire;
        delete _pwm;
        delete _servos;
        delete _config;
    }

    void Env::_sleep(double rate)
    {
        int milli = static_cast<int>(1000.0 / rate);
        std::this_thread::sleep_for(std::chrono::milliseconds(milli));
    }

    void Env::_syncEnv(double rate)
    {
        _sleep(rate);

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
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("cannot update event data");
        }
    }

    bool Env::isDone()
    {

        // return true;
        bool done = true;
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            if (_disableServo[servo]) {
                continue;
            }
            else {
                done = _currentData[servo].done;
            }
        } 

        return done;
    }

    void Env::_resetEnv()
    {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            _servos->setAngle(servo, _resetAngles[servo]);
            _lastAngles[servo] = _resetAngles[servo];
            _currentAngles[servo] = _resetAngles[servo];
            _pids[servo]->init();
            
            for (int i = 0; i < 5; i++) {
                _errors[servo][i] = 0.0;
                _outputs[servo][i] = 0.0;
            }
        }
    }

    Utility::RD Env::reset()
    {
        _resetEnv();
        _syncEnv();

        Utility::RD data = {};

        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            data.servos[servo].pidStateData = _pids[servo]->getState(true);
            data.servos[servo].obj =  _currentData[servo].obj;
            data.servos[servo].frame = _currentData[servo].frame;
            data.servos[servo].lastAngle = Utility::mapOutput(_lastAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            data.servos[servo].currentAngle = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            data.servos[servo].spf = _currentData[servo].spf;
    

            for (int i = 0; i < 5; i++) {
                _outputs[servo][i] = 0.0;
                _errors[servo][i] = 0.0;
            }

            double frameCenter = (_config->dims[servo] / 2.0);
            _outputs[servo][0] = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            _errors[servo][0] = (_invert[servo]) ? (frameCenter - _currentData[servo].obj) / frameCenter : (_currentData[servo].obj - frameCenter) / frameCenter;
            // _errors[servo][0] = (_currentData[servo].obj - frameCenter) / frameCenter;
            data.servos[servo].setData(_errors[servo], _outputs[servo]);
            _stateData[servo] = data.servos[servo];
            _predObjLoc[servo] = 0.0;
        }

        return data;
    }

    // Using action, take step and return observation, reward, done, and actions for every servo. 
    // Note: SR[servo].currentState is always null. Retrieve currentState from previous 'step' or 'reset' call.
    Utility::SR Env::step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale, double rate)
    {
        _currentSteps = (_currentSteps + 1) % INT_MAX; 
        double rescaledActions[NUM_SERVOS][NUM_ACTIONS];

        Utility::SR stepResults;
        
        for (int servo = 0; servo < NUM_SERVOS; servo++) {

            if (_disableServo[servo]) {
                continue;
            }

            if (!_config->usePIDs) {

                // Derive output angle and object's next location from SAC output actions
                stepResults.servos[servo].actions[0] = actions[servo][0];
                stepResults.servos[servo].actions[1] = actions[servo][1];

                if (rescale) {
                    rescaledActions[servo][0] = Utility::rescaleAction(actions[servo][0], _config->actionLow, _config->actionHigh); // Angle
                    rescaledActions[servo][1] = Utility::rescaleAction(actions[servo][1], 0, _config->dims[servo]); // Estimate of location
                    _predObjLoc[servo] = rescaledActions[servo][1];
                } else {
                    for (int a = 0; a < NUM_ACTIONS; a++) {
                        rescaledActions[servo][a] = actions[servo][a];
                    }  
                }
            } else {

                // Derive PID gains from SAC output actions
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    stepResults.servos[servo].actions[a] = actions[servo][a];

                    if (rescale) {
                        rescaledActions[servo][a] = Utility::rescaleAction(actions[servo][a], _config->actionLow, _config->actionHigh);
                    } else {
                        rescaledActions[servo][a] = actions[servo][a];
                    }
                }
            }
            

            double newAngle = 0.0;
            if (!_config->usePIDs) {
                newAngle = rescaledActions[servo][0];
            }
            else {
                _pids[servo]->setWeights(rescaledActions[servo][0], rescaledActions[servo][1], rescaledActions[servo][2]);
                newAngle = _pids[servo]->update(_currentData[servo].obj);
            }

            // if (_invert[servo]) { 
            //     newAngle = newAngle * -1.0; 
            // }

            _lastAngles[servo] = _currentAngles[servo];
            _currentAngles[servo] = newAngle;
            _servos->setAngle(servo, newAngle);
        }
        
        _syncEnv(rate);        

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
                stepResults.servos[servo].errors[0] = stepResults.servos[servo].reward;
                stepResults.servos[servo].errors[1] = Utility::predictedObjectLocationToReward(rescaledActions[servo][1], _currentData[servo].obj, static_cast<double>(_config->dims[servo]), _currentData[servo].done);;
                stepResults.servos[servo].reward += stepResults.servos[servo].errors[1];
            }

            // Update state information
            stepResults.servos[servo].nextState.obj = _currentData[servo].obj;
            stepResults.servos[servo].nextState.frame = _currentData[servo].frame;
            stepResults.servos[servo].nextState.lastAngle = Utility::mapOutput(_lastAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            stepResults.servos[servo].nextState.currentAngle = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            stepResults.servos[servo].nextState.spf = _currentData[servo].spf;
            stepResults.servos[servo].done = _currentData[servo].done;
            stepResults.servos[servo].empty = false;

            _stateData[servo] = stepResults.servos[servo].nextState;

            // Store error and output history
            for (int i = 4; i >= 0; i--) {
                _outputs[servo][i + 1] = _outputs[servo][i];
                _errors[servo][i + 1] = _errors[servo][i];
            }

            // Scale to -1 to 1;
            double frameCenter = (_config->dims[servo] / 2.0);
            _outputs[servo][0] = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
            _errors[servo][0] = (_invert[servo]) ? (frameCenter - _currentData[servo].obj) / frameCenter : (_currentData[servo].obj - frameCenter) / frameCenter;
            
            // Fill out the step results
            if (!_config->usePIDs) {
                
                // We still use PIDS for keeping track of error in ENV. Otherwise the caller supplies actions rather than PIDS
                _pids[servo]->update(currentError);
                stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(true);
                stepResults.servos[servo].nextState.setData(_errors[servo], _outputs[servo]);
            }
            else {
                stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(true);
            }
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

    void Env::getCurrentState(Utility::SD state[NUM_SERVOS]) {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            state[servo] = _stateData[servo];
        }
    }

    void Env::getPredictedObjectLocation(double locations[NUM_SERVOS]) {
        std::unique_lock<std::mutex> lck(_lock);
        try
        {
            for (int servo = 0; servo < NUM_SERVOS; servo++) {
                if (_disableServo[servo]) {
                    continue;
                }

                locations[servo] = _predObjLoc[servo];
            }
            
            lck.unlock();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            throw std::runtime_error("cannot get current object locations");
        }
    }
};