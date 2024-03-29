#include "Env.h"

Env::Env(control::ServoKit* servos, Utility::Config* config) {
    _eventData;
    _lastData;
    _currentData;
    _servos = servos;

    // Setup default env state variables
    _config = config;
    _currentSteps = 0;
    _maxPreSteps = 2;
    _preStepAngleAmount = 0.1;
    _errorThreshold  = 0.005;
    _updated = false;
    _start = std::chrono::steady_clock::now();
    _actionType = _config->actionType;

    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        double setpoint = static_cast<double>(_config->dims[servo]) / 2.0;
        _preSteps[servo] = 0;
        _recentReset[servo] = true;
        _pids[servo] = new PID(	_config->defaultGains[0], 
                                _config->defaultGains[1], 
                                _config->defaultGains[2], 
                                _config->pidOutputLow, 
                                _config->pidOutputHigh, 
                                setpoint);
        
        _resetAngles[servo] = _config->resetAngles[servo];
        _disableServo[servo] = _config->disableServo[servo];
        _invertData[servo] = _config->invertData[servo];
        _invertAngles[servo] = _config->invertAngles[servo];
        _currentAngles[servo] = _config->resetAngles[servo];
        _lastAngles[servo] = _config->resetAngles[servo];
        _currentDeltaAngles[servo] = 0.0;

        _servos->initServo(_config->servoConfigurations[servo]);

        for (int i = 0; i < HISTORY_BUFFER_SIZE; i++) {
            _outputs[servo][i] = 0.0;
            _errors[servo][i] = 0.0;
            _angleTimestamps[servo][i] = 0.0;
            _errorTimestamps[servo][i] = 0.0;
        }
    }
}

Env::~Env() {

}

void Env::_sleep(double rate) {
    int milli = static_cast<int>(1000.0 / rate);
    std::this_thread::sleep_for(std::chrono::milliseconds(milli));
}

void Env::_syncEnv(double rate) {
    _sleep(rate);

    std::unique_lock<std::mutex> lck(_dataLock);

    // Sleep for specified time and Wait for env to respond to changes
    try {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            
            if (_disableServo[servo]) {
                continue;
            }				

            while (_eventData[servo].timestamp == _lastTimeStamp[servo]) {
                _dataCondition.wait(lck);
            }

            _lastData[servo] = _currentData[servo];
            _currentData[servo] = _eventData[servo];
            _lastTimeStamp[servo] = _currentData[servo].timestamp;
        }

        lck.unlock();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("cannot sync with servos");
    }
}

void Env::waitForUpdate() {
    std::unique_lock<std::mutex> lck(_updateLock);
    while (!_updated) {
        _updateCondition.wait(lck);
    }
    lck.unlock();
}

void Env::update(Utility::ED eventDataArray[NUM_SERVOS]) {
    
    std::unique_lock<std::mutex> lck(_dataLock);
    try {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            if (_disableServo[servo]) {
                continue;
            }

            this->_eventData[servo] = eventDataArray[servo];
        }
        
        lck.unlock();
        _dataCondition.notify_all();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("cannot update event data");
    }
}

bool Env::isDone() {
    bool done = true;
    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        if (_disableServo[servo]) {
            continue;
        } else {
            done = _currentData[servo].done;
        }
    } 

    return done;
}

void Env::_resetEnv(bool overrideResetAngles, double angles[NUM_SERVOS]) {
    
    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        _recentReset[servo] = true;
        _preSteps[servo] = 0;
        if (_disableServo[servo]) {
            continue;
        }
        
        double newAngle = overrideResetAngles ? angles[servo] : _resetAngles[servo];
        _servos->setAngle(servo, (_invertAngles[servo]) ? -newAngle : newAngle);

        _currentDeltaAngles[servo] = 0.0;
        _lastAngles[servo] =  _resetAngles[servo];
        _currentAngles[servo] = overrideResetAngles ? angles[servo] : _resetAngles[servo];
        _pids[servo]->init();
        
        for (int i = 0; i < HISTORY_BUFFER_SIZE; i++) {
            _errors[servo][i] = 0.0;
            _outputs[servo][i] = 0.0;
            _deltaAngles[servo][i] = 0.0;
            _angleTimestamps[servo][i] = 0.0;
            _errorTimestamps[servo][i] = 0.0;
        }
    }
}

Utility::RD Env::reset(bool useCurrentAngles) {
    // std::unique_lock<std::mutex> lck(_updateLock);
    _updated = false;
    _start = std::chrono::steady_clock::now();
    _resetEnv(useCurrentAngles, _currentAngles);
    _syncEnv();

    Utility::RD data = {};
    data.setActionType(_actionType);

    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        if (_disableServo[servo]) {
            continue;
        }
        
        // TODO::Cleanup old and unused state variables
        data.servos[servo].pidStateData = _pids[servo]->getState(true);
        data.servos[servo].currentAngle = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
        data.servos[servo].spf = _currentData[servo].spf;
        data.servos[servo].tracking = _currentData[servo].tracking;

        int frameCenter = _config->dims[servo] / 2;
        _outputs[servo][0] = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo], _config->anglesHigh[servo], 0.0, 1.0);
        _errors[servo][0] = Utility::mapOutput(_invertData[servo] ? static_cast<double>(frameCenter - static_cast<int>(_currentData[servo].obj)) :  static_cast<double>(static_cast<int>(_currentData[servo].obj) - frameCenter), - static_cast<double>(frameCenter),  static_cast<double>(frameCenter), 0.0, 1.0);
        _angleTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        _errorTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        _deltaAngles[servo][0] = 0.0;
        data.servos[servo].setData(
            _errors[servo], 
            _outputs[servo], 
            _deltaAngles[servo], 
            _angleTimestamps[servo],
            _errorTimestamps[servo]
        );
        _stateData[servo] = data.servos[servo];
        _currentDeltaAngles[servo] = 0.0;
        _predObjLoc[servo] = 0.0;
    }

    _updated = true;
    // lck.unlock();
    // _updateCondition.notify_all();
    return data;
}

Utility::RD Env::reset(double angles[NUM_SERVOS]) {
    // std::unique_lock<std::mutex> lck(_updateLock);
    _updated = false;
    _start = std::chrono::steady_clock::now();
    _resetEnv(true, angles);
    _syncEnv();

    Utility::RD data = {};
    data.setActionType(_actionType);

    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        if (_disableServo[servo]) {
            continue;
        }

        data.servos[servo].pidStateData = _pids[servo]->getState(true);
        data.servos[servo].currentAngle = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
        data.servos[servo].spf = _currentData[servo].spf;
        data.servos[servo].tracking = _currentData[servo].tracking;

        int frameCenter = _config->dims[servo] / 2;
        _outputs[servo][0] = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo], _config->anglesHigh[servo], 0.0, 1.0);
        _errors[servo][0] = Utility::mapOutput(_invertData[servo] ? static_cast<double>(frameCenter - static_cast<int>(_currentData[servo].obj)) :  static_cast<double>(static_cast<int>(_currentData[servo].obj) - frameCenter), - static_cast<double>(frameCenter),  static_cast<double>(frameCenter), 0.0, 1.0);
        _angleTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        _errorTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        _deltaAngles[servo][0] = 0.0;
        data.servos[servo].setData(
            _errors[servo], 
            _outputs[servo], 
            _deltaAngles[servo], 
            _angleTimestamps[servo],
            _errorTimestamps[servo]
        );
        _stateData[servo] = data.servos[servo];
        _currentDeltaAngles[servo] = 0.0;
        _predObjLoc[servo] = 0.0; 
    }

    _updated = true;
    // lck.unlock();
    // _updateCondition.notify_all();
    return data;
}

// Using action, take step and return observation, reward, done, and actions for every servo. 
// Note: SR[servo].currentState is always null. Retrieve currentState from previous 'step' or 'reset' call.
Utility::SR Env::step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale, double rate) {
    
    // std::unique_lock<std::mutex> lck(_updateLock);
    // Update/set various loop variants
    _currentSteps = (_currentSteps + 1) % INT_MAX; 
    _updated = false;
    double rescaledActions[NUM_SERVOS][NUM_ACTIONS];
    double maxDeltaAngle = _config->maxDeltaAngle;
    bool empty[NUM_SERVOS] = { false };
    
    // Return value
    Utility::SR stepResults = {};
    stepResults.setActionType(_actionType);

    // Delta time 
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    double dt = std::clamp<double>(std::chrono::duration<double>(now - _start).count(), 0.0, 1.0);
    _start = std::chrono::steady_clock::now();
     
    // Foreach servo, calculate predicted object locations and new servo angles
    for (int servo = 0; servo < NUM_SERVOS; servo++) {

        if (_disableServo[servo]) {
            continue;
        }

        switch (_config->actionType)
        {
        case Utility::ActionType::PID:
            // Derive PID gains from SAC output actions
            for (int a = 0; a < NUM_ACTIONS; a++) {
                stepResults.servos[servo].actions[a] = actions[servo][a];

                if (rescale) {
                    rescaledActions[servo][a] = Utility::rescaleAction(actions[servo][a], _config->actionLow, _config->actionHigh);
                } else {
                    rescaledActions[servo][a] = actions[servo][a];
                }
            }
            break;  
        case Utility::ActionType::ANGLE:
            if (!_recentReset[servo]) { 
                _preSteps[servo] = 0;
                
                // Derive output angle and object's next location from SAC output actions
                stepResults.servos[servo].actions[0] = actions[servo][0];

                if (_config->usePOT) {
                    stepResults.servos[servo].actions[1] = actions[servo][1];
                }

                if (rescale) {               
                    rescaledActions[servo][0] = Utility::rescaleAction(actions[servo][0], _config->actionLow, _config->actionHigh); // Angle

                    if (_config->usePOT) {
                        if (_invertData[servo]) {
                            rescaledActions[servo][1] = Utility::rescaleAction(-actions[servo][1], 0, _config->dims[servo]); 
                            
                        } else {
                            rescaledActions[servo][1] = Utility::rescaleAction(actions[servo][1], 0, _config->dims[servo]);
                        }
                        
                        _predObjLoc[servo] = std::round(rescaledActions[servo][1]); // Pixel location cannot have arbitrary precision
                    }     
                } else {
                    for (int a = 0; a < NUM_ACTIONS; a++) {
                        rescaledActions[servo][a] = actions[servo][a];
                    }  
                }
            } else {
                /* 
                
                    Three datapoints should allow for far better movement by an AI, in exchange for this small observation delay.
                    Move a small amount now, preventing a blind over-shot by the AI (Equivelent to dirivitive kick in PID controllers).
                    This dirivitive kick is better minimized by AI with more datapoints. This is due to the fact that it is impossible
                    to determine speed (two datapoints) or acceleration (three datapoints) with a single datapoint in a dynamic system.

                    Formulas for first and second order delta error: 
                    1.) (E(t_i) - E(t_i + 1)) / T == Speed
                    2.) (E(t_i) - ( 2.0 * E(t_i + 1) ) + E(t_i + 2)) / T^2 == Acceleration
                    
                */
                
                if (_preSteps[servo] >= _maxPreSteps) {
                    _recentReset[servo] = false;
                    _preSteps[servo] = 0;
                } else {
                    _preSteps[servo]++;
                }
                
                empty[servo] = true;
                int frameCenter = _config->dims[servo] / 2;
                double error = _invertData[servo] ? static_cast<double>(frameCenter - static_cast<int>(_currentData[servo].obj)) : static_cast<double>(static_cast<int>(_currentData[servo].obj) - frameCenter)  / static_cast<double>(frameCenter);

                // If error is negligible, then dont move at all
                if (std::abs<double>(error) <= _errorThreshold || _preStepAngleAmount == 0.0) {
                    rescaledActions[servo][0] = _currentAngles[servo];
                } else {
                    // inverting if needed, while enforcing angle bounds. 
                    if (Utility::calculateDirectionOfObject(error, _invertData[servo])) {
                        rescaledActions[servo][0] = std::clamp<double>(_currentAngles[servo] - _preStepAngleAmount, _config->anglesLow[servo], _config->anglesHigh[servo]);
                    } else {
                        rescaledActions[servo][0] = std::clamp<double>(_currentAngles[servo] + _preStepAngleAmount, _config->anglesLow[servo], _config->anglesHigh[servo]);
                    }
                } 

                // Provide anyways
                stepResults.servos[servo].actions[0] = Utility::mapOutput(rescaledActions[servo][0], _config->anglesLow[servo], _config->anglesHigh[servo], -1.0, 1.0);
                if (_config->usePOT) {
                    stepResults.servos[servo].actions[1] = 0.0;
                }
            }
            break;
        case Utility::ActionType::SPEED:

            if (!_recentReset[servo]) { 
                _preSteps[servo] = 0;
                
                // Derive output angles as a function of speed and time
                double throttle = actions[servo][0];
                double minSpeed = _config->minServoSpeed;
                double maxAnglesPerSecond = 60.0 / minSpeed;
                
                // Clip deltaAngle and the outputAngle
                double newAngle = std::clamp<double>(maxAnglesPerSecond * throttle * dt, -maxDeltaAngle, maxDeltaAngle) + _currentAngles[servo];
                newAngle = std::clamp<double>(newAngle, _config->anglesLow[servo], _config->anglesHigh[servo]);
                rescaledActions[servo][0] = newAngle;
                
                // Hold onto actions for step results
                stepResults.servos[servo].actions[0] = actions[servo][0];
                if (_config->usePOT) {
                    stepResults.servos[servo].actions[1] = actions[servo][1];
                }

                _currentDeltaAngles[servo] = Utility::mapOutput((rescaledActions[servo][0] - _currentAngles[servo]) / maxDeltaAngle, -1.0, 1.0, 0.0, 1.0);
             
                if (_config->usePOT) {
                    if (_invertData[servo]) {
                        rescaledActions[servo][1] = Utility::rescaleAction(-actions[servo][1], 0, _config->dims[servo]); 
                        
                    } else {
                        rescaledActions[servo][1] = Utility::rescaleAction(actions[servo][1], 0, _config->dims[servo]);
                    }
                    
                    _predObjLoc[servo] = std::round(rescaledActions[servo][1]); // Pixel location cannot have arbitrary precision
                }
            } else {
                
                if (_preSteps[servo] >= _maxPreSteps) {
                    _recentReset[servo] = false;
                    _preSteps[servo] = 0;
                } else {
                    _preSteps[servo]++;
                }
                
                // rescaledActions[servo][0] = _currentAngles[servo];
                // _currentDeltaAngles[servo] = 0.0;
                // Step in direction of object's motion
                empty[servo] = true;
                int frameCenter = _config->dims[servo] / 2;
                double error = _invertData[servo] ? static_cast<double>(frameCenter - static_cast<int>(_currentData[servo].obj)) : static_cast<double>(static_cast<int>(_currentData[servo].obj) - frameCenter)  / static_cast<double>(frameCenter);

                // If error is negligible, then dont move at all
                if (std::abs<double>(error) <= _errorThreshold || _preStepAngleAmount == 0.0) {
                    rescaledActions[servo][0] = _currentAngles[servo];
                } else {
                    // inverting if needed, while enforcing angle bounds. 
                    if (Utility::calculateDirectionOfObject(error, _invertData[servo])) {
                        rescaledActions[servo][0] = std::clamp<double>(_currentAngles[servo] - _preStepAngleAmount, _config->anglesLow[servo], _config->anglesHigh[servo]);
                    } else {
                        rescaledActions[servo][0] = std::clamp<double>(_currentAngles[servo] + _preStepAngleAmount, _config->anglesLow[servo], _config->anglesHigh[servo]);
                    }
                } 

                _currentDeltaAngles[servo] = Utility::mapOutput((rescaledActions[servo][0] - _currentAngles[servo]) / maxDeltaAngle, -1.0, 1.0, 0.0, 1.0);

                // Provide anyways
                stepResults.servos[servo].actions[0] = Utility::mapOutput(rescaledActions[servo][0], _config->anglesLow[servo], _config->anglesHigh[servo], -1.0, 1.0);;
                if (_config->usePOT) {
                    stepResults.servos[servo].actions[1] = 0.0;
                }
            }
            break;

        default:
            throw std::runtime_error("Invalid action type configured in config");
        }
        
        double newAngle = 0.0;
        switch (_config->actionType)
        {
        case Utility::ActionType::PID:
            _pids[servo]->setWeights(rescaledActions[servo][0], rescaledActions[servo][1], rescaledActions[servo][2]);
            newAngle = _pids[servo]->update(_currentData[servo].obj, _invertData[servo]);
            break;
        
        case Utility::ActionType::ANGLE:
            newAngle = rescaledActions[servo][0];
            break;

        case Utility::ActionType::SPEED:
            newAngle = rescaledActions[servo][0];
            break;

        default:
            throw std::runtime_error("Invalid action type configured in config");
        }


        _lastAngles[servo] = _currentAngles[servo];
        _currentAngles[servo] = newAngle;
        _servos->setAngle(servo, (_invertAngles[servo]) ? -newAngle : newAngle);
    }
    
    _syncEnv(rate);        

    for (int servo = 0; servo < NUM_SERVOS; servo++) {

        if (_disableServo[servo]) {
            continue;
        }		

        double lastError = _lastData[servo].obj;
        double currentError = _currentData[servo].obj;
        empty[servo] = !static_cast<bool>(_currentData[servo].tracking);

        // Calculate rewards
        if (_lastData[servo].done) {
            throw std::runtime_error("State must represent a complete transition");
        }
        else {
            if (empty[servo]) {
                stepResults.servos[servo].reward = 0.0;
            } else if (static_cast<bool>(_lastData[servo].tracking)) {
                stepResults.servos[servo].reward = Utility::pidErrorToReward(static_cast<int>(currentError), static_cast<int>(lastError), _config->dims[servo] / 2, _currentData[servo].done, false, 0.0);
            } else {
                stepResults.servos[servo].reward = -currentError / static_cast<double>(_config->dims[servo]);
            }
            
            stepResults.servos[servo].errors[0] = stepResults.servos[servo].reward;
            if (_config->usePOT) {
                stepResults.servos[servo].errors[1] = Utility::predictedObjectLocationToReward(static_cast<int>(rescaledActions[servo][1]), static_cast<int>(_currentData[servo].obj), _config->dims[servo], _currentData[servo].done, false, 0.0);
                stepResults.servos[servo].reward += stepResults.servos[servo].errors[1];
            }
        }

        // Update state information
        stepResults.servos[servo].nextState.currentAngle = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], -1.0, 1.0);
        stepResults.servos[servo].nextState.spf = _currentData[servo].spf;
        stepResults.servos[servo].nextState.tracking = _currentData[servo].tracking;
        stepResults.servos[servo].done = _currentData[servo].done;
        stepResults.servos[servo].empty = empty[servo];
        _stateData[servo] = stepResults.servos[servo].nextState;

        // Store output and deltaAngle history
        int frameCenter = _config->dims[servo] / 2;
        for (int i = HISTORY_BUFFER_SIZE - 2; i >= 0; i--) {
            _outputs[servo][i + 1] = _outputs[servo][i];
            _deltaAngles[servo][i + 1] = _deltaAngles[servo][i];
            _angleTimestamps[servo][i + 1] = _angleTimestamps[servo][i];
        }
        _outputs[servo][0] = Utility::mapOutput(_currentAngles[servo], _config->anglesLow[servo],  _config->anglesHigh[servo], 0.0, 1.0);
        _deltaAngles[servo][0] = _currentDeltaAngles[servo];
        _angleTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        
        // Store err if tracking
        if (static_cast<bool>(_currentData[servo].tracking)) {
            for (int i = HISTORY_BUFFER_SIZE - 2; i >= 0; i--) {
                _errors[servo][i + 1] = _errors[servo][i];
                _errorTimestamps[servo][i + 1] = _errorTimestamps[servo][i]; 
            }
            _errors[servo][0] = Utility::mapOutput(_invertData[servo] ? static_cast<double>(frameCenter - static_cast<int>(_currentData[servo].obj)) : static_cast<double>(static_cast<int>(_currentData[servo].obj) - frameCenter), -static_cast<double>(frameCenter), static_cast<double>(frameCenter), 0.0, 1.0);
            _errorTimestamps[servo][0] = Utility::removeWhole(_currentData[servo].timestamp);
        }

        switch (_config->actionType)
        {
        case Utility::ActionType::PID:
            stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(true);
            break;
        case Utility::ActionType::ANGLE:
            _pids[servo]->update(currentError);
            stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(true);
            stepResults.servos[servo].nextState.setData(
                _errors[servo], 
                _outputs[servo], 
                _deltaAngles[servo], 
                _angleTimestamps[servo],
                _errorTimestamps[servo]
            );
            break;
        case Utility::ActionType::SPEED:
            _pids[servo]->update(currentError);
            stepResults.servos[servo].nextState.pidStateData = _pids[servo]->getState(true);
            stepResults.servos[servo].nextState.setData(
                _errors[servo], 
                _outputs[servo], 
                _deltaAngles[servo], 
                _angleTimestamps[servo],
                _errorTimestamps[servo]
            );
            break;
        default:
            throw std::runtime_error("Invalid action type configured in config");
        }
    }
    
    _updated = true;
    // lck.unlock();
    // _updateCondition.notify_all();
    
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

void Env::getCurrentAngle(double angles[NUM_SERVOS]) {
    std::unique_lock<std::mutex> lck(_dataLock);
    try
    {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            if (_disableServo[servo]) {
                continue;
            }

            angles[servo] = _currentAngles[servo];
        }
        
        lck.unlock();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("cannot get current servo angles");
    }
}

void Env::getPredictedObjectLocation(double locations[NUM_SERVOS]) {
    std::unique_lock<std::mutex> lck(_dataLock);
    try {
        for (int servo = 0; servo < NUM_SERVOS; servo++) {
            if (_disableServo[servo]) {
                continue;
            }
            
            locations[servo] = _predObjLoc[servo];            
        }
        
        lck.unlock();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("cannot get current object locations");
    }
}