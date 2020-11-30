
#pragma once

#include "ServoEnv.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "../util/data.h"
#include "../wire/Wire.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"
#include "../network/SACAgent.h"
#include "../network/ExpReplayBuffer.h"

/*
    This is a base class only
*/

namespace TATS {

    class PanTiltManager 
    {
    private:
        ServoEnv* _pan;
        ServoEnv* _tilt;

        control::Wire* _wire;
        control::PCA9685* _pwm;
        control::ServoKit* _servos;

        Utility::cfg* _config;

        SACAgent* _panAutoTuner;
        SACAgent* _tiltAutoTuner;

        ExpReplayBuffer* _panBuff;
        ExpReplayBuffer* _tiltBuff;
        

    public:
        PanTiltManager();
        ~PanTiltManager();

        void update(EventData pan, EventData tilt);
        std::vector<Transition> step(Actions pan, Actions tilt);
        std::vector<bool> done();

        
    };
};
