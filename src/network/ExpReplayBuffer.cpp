#include "ExpReplayBuffer.h"

#include "../util/util.h"
#include "../util/data.h"

namespace TATS {
    ExpReplayBuffer::ExpReplayBuffer(int maxBufferSize)
    {
        _maxBufferSize = maxBufferSize;
    }

    ExpReplayBuffer::~ExpReplayBuffer() {

    }

    int ExpReplayBuffer::_draw(int min, int max)
    {
        return std::uniform_int_distribution<int>{min, max}(eng);
    }

    std::vector<Transition> ExpReplayBuffer::sample(int batchSize, int startingIndex)
    {
        std::vector<Transition> batch;

        if (batchSize > _trainingBuffer.size()) {
            throw std::runtime_error("Batch size cannot be larger than buffer size");
        }

        std::unique_lock<std::mutex> lck(_lock);

        for (int i = 0; i < batchSize; i++) {
            int number = _draw(startingIndex, _trainingBuffer.size() - 1);

            batch.push_back(_trainingBuffer.at(number));
        }

        lck.unlock();
    }


    void ExpReplayBuffer::add(Transition data)
    {
        std::unique_lock<std::mutex> lck(_lock);
        
        if (_trainingBuffer.size() == _maxBufferSize) {
            _trainingBuffer.erase(_trainingBuffer.begin());	
            _trainingBuffer.push_back(data);
        }
        else {
            _trainingBuffer.push_back(data);
        }

        lck.unlock();
    }

    int ExpReplayBuffer::size() {   
        std::unique_lock<std::mutex> lck(_lock);
        int size = _trainingBuffer.size();
        lck.unlock();
        return size;
    }

    void ExpReplayBuffer::clear() {
        std::unique_lock<std::mutex> lck(_lock);
        _trainingBuffer.clear();
        lck.unlock();
    }
};
