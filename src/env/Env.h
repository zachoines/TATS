#pragma once
#include <cmath>
#include <random>
#include <condition_variable>
#include <mutex>
#include <bits/stdc++.h>

#include "../util/config.h"
#include "../util/util.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"
#include "../wire/Wire.h"
#include "../pid/PID.h"
class Env
{
private:

    std::mutex _dataLock;
    std::condition_variable _dataCondition;
    std::mutex _updateLock;
    std::condition_variable _updateCondition;
    Utility::cfg* _config;
    control::ServoKit* _servos;

    int _frameSkip;
    double _lastTimeStamp[NUM_SERVOS];

    Utility::ED _eventData[NUM_SERVOS] = {{}};
    Utility::ED _lastData[NUM_SERVOS] = {{}};
    Utility::ED _currentData[NUM_SERVOS] = {{}};
    Utility::SD _stateData[NUM_SERVOS] = {{}};

    bool _updated;
    bool _invertData[NUM_SERVOS];
    bool _invertAngles[NUM_SERVOS];
    bool _disableServo[NUM_SERVOS];
    double _resetAngles[NUM_SERVOS];
    double _currentAngles[NUM_SERVOS] = { 0.0 };
    double _lastAngles[NUM_SERVOS] = { 0.0 };
    double _errors[NUM_SERVOS][ERROR_LIST_SIZE] = { 0.0 };
    double _outputs[NUM_SERVOS][ERROR_LIST_SIZE] = { 0.0 };
    double _predObjLoc[NUM_SERVOS] = { 0.0 };
    int _currentSteps; 
    bool _recentReset[NUM_SERVOS];
    int _preSteps[NUM_SERVOS];
    int _maxPreSteps;
    int _preStepAngleAmount;
    double _errorThreshold;

    PID* _pids[NUM_SERVOS];

    /***
     * @brief Sleeps this thread, 1 / rate 
     * @param rate Fraction of second, 1 / rate
     * @return nothing
     */
    void _sleep(double rate=10);

    /***
     * @brief Sleeps for 1 / rate, then waits for signal that new data is available
     * @param rate Fraction of second, 1 / rate
     * @return nothing
     */
    void _syncEnv(double rate=10);

    /***
     * @brief Set internal data and servos to defaults, with option to override angles
     * @param overrideResetAngles Override and default reset angles
     * @param angles Array with angles to reset env with 
     * @return nothing
     */
    void _resetEnv(bool overrideResetAngles, double angles[NUM_SERVOS]); // Resets servos and re-inits PID's. Call only once manually.

public:
    Env(control::ServoKit* servos);
    ~Env();

    /***
     * @brief Returns true if the target is lost
     * @return a boolean
     */
    bool isDone();

    /***
     * @brief Sets which servos are currently disabled
     * @param servos Boolean array setting active servos 
     * @return nothing
     */
    void setDisabled(bool servos[NUM_SERVOS]);

    /***
     * @brief Indicates which servos are currently disabled
     * @param servos Boolean array indicating active servos
     * @return nothing
     */
    void getDisabled(bool servos[NUM_SERVOS]);

    /***
     * @brief Reset the env and waits for the environment to update
     * @param useCurrentAngles Whether to use angle of servos
     * @return Current state data
     */
    Utility::RD reset(bool useCurrentAngles=false);  // 

    /***
     * @brief Reset the env and waits for the environment to update
     * @param angles Override default angles with provided angles
     * @return Current state data
     */
    Utility::RD reset(double angles[NUM_SERVOS]); 


    /***
     * @brief Steps with actions and waits for Env to update, then returns the current state.
     * @param actions 2D array containing the desired action foreach servo
     * @param rescale rescale actions to range defined in config.h
     * @param rate sleeps for 1 / rate
     * @return current state of env
     */
    Utility::SR step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale = true, double rate = 5);  

    /***
     * @brief Called from another thread to update state variables
     * @param eventDataArray Event data to update env with 
     * @return void
     */
    void update(Utility::ED eventDataArray[NUM_SERVOS]);

    /***
     * @brief Called from another thread. Blocks until servos update.
     * @return void
     */
    void waitForUpdate();

    /***
     * @brief Retrieves current state data of env
     * @param state Filled with state data
     * @return void
     */
    void getCurrentState(Utility::SD state[NUM_SERVOS]);

    /***
     * @brief Retrieves predicted object location stored from previous call to "step"
     * @param locations Fills the array with predicted object locations
     * @return void
     */
    void getPredictedObjectLocation(double locations[NUM_SERVOS]);

    /***
     * @brief Get the current angle of the servos
     * @param angles Fills the array with current angles
     * @return void
     */
    void getCurrentAngle(double angles[NUM_SERVOS]);
};
