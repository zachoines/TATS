#include "PanTiltManager.h"

namespace TATS {
    
    PanTiltManager::PanTiltManager() {
        // Setup pan/tilt, I2C, and PWM
        _wire = new control::Wire();
        _pwm = new control::PCA9685(0x40, _wire);
        _servos = new control::ServoKit(_pwm);
        _tilt = new ServoEnv(_servos, 0);
        _pan = new ServoEnv(_servos, 1);
        _config = new Utility::Config();

        _panBuff = new ExpReplayBuffer(_config->maxBufferSize);
        _tiltBuff = new ExpReplayBuffer(_config->maxBufferSize);

    }

    PanTiltManager::~PanTiltManager() {
        delete _wire;
        delete _pwm;
        delete _servos;
        delete _tilt;
        delete _pan;
    }

    void PanTiltManager::update(Actions pan, Actions tilt) {
        return;
    }
    
    std::vector<Transition> PanTiltManager::step() {
        Transition data;
        return { data };
    }
    
    std::vector<bool> PanTiltManager::done() {
        return { false, false };
    }
};