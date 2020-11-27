#pragma once

#include <vector>
#include "../util/data.h"
#include "../wire/Wire.h"
#include "../pid/PID.h"

namespace TATS {

    typedef std::vector<double> StateData;
    typedef std::vector<double> Actions;
    typedef std::vector<double> EventData;
    typedef std::vector<int> MetaData;

    struct TransitionData {
        StateData currentState;
        StateData nextState;
        Actions actions;
        MetaData meta;
        double reward;
        bool done;
        bool empty;

        TransitionData(int actionSize=1, int stateSize=10) :
            currentState(stateSize, 0.0),
            nextState(stateSize, 0.0),
            actions(actionSize, 0.0),
            meta(stateSize, 0),
            reward(0.0),
            done(false),
            empty(true)
        {}

    } typedef Transition;

    class EnvBase
    {
    private:

        virtual void _syncEnv() = 0;
        virtual void _resetEnv() = 0;

    public:
        EnvBase();
        ~EnvBase();

        virtual bool isDone() = 0;
        virtual StateData reset() = 0; 
        virtual Transition step(Actions actions, bool rescale = true) = 0;
        virtual void update(EventData state, double timeStamp) = 0; 
    };
};
