#pragma once
#include <vector>
#include <random>
#include <iterator>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "../util/util.h"
#include "../util/data.h"

#include <mutex>

namespace TATS {
    class ExpReplayBuffer
    {
    private:
        std::vector<TATS::Transition> _trainingBuffer;
        int _maxBufferSize;
        bool _multiprocess;

        // Create a random device and use it to generate a random seed
        std::mt19937 eng{ std::random_device{}() };
        std::mutex _lock;
        
        int _draw(int min, int max);

    public:
        ExpReplayBuffer(int maxBufferSize);
        ~ExpReplayBuffer();
        std::vector<TATS::Transition> sample(int batchSize = 32, int startingIndex = 0);
        void add(TATS::Transition data);
        int size();
        void clear();
    };
};


