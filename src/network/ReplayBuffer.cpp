#include "ReplayBuffer.h"
#include <vector>
#include <random>
#include <iterator>

#include <cstdlib>
#include <ctime>
#include <iostream>


#include "../util/util.h"
#include "../util/config.h"


ReplayBuffer::ReplayBuffer(int maxBufferSize, Utility::SharedBuffer* buffer, bool multiprocess) {
    _trainingBuffer = buffer;
    _mutex = new boost::interprocess::named_mutex(boost::interprocess::open_or_create, "ReplayBufferMutex");
    _mutex->unlock();
    _maxBufferSize = maxBufferSize;
    _multiprocess = multiprocess;
}

ReplayBuffer::~ReplayBuffer() {
    if (_multiprocess) {
        boost::interprocess::named_mutex::remove("ReplayBufferMutex");
        delete _mutex;
    }
}

int ReplayBuffer::_draw(int min, int max) {
    return std::uniform_int_distribution<int>{min, max}(eng);
}

Utility::TrainBuffer ReplayBuffer::ere_sample(int batchSize, int startingIndex) {
    Utility::TrainBuffer batch;

    if (_multiprocess) {
        if (batchSize > _trainingBuffer->size()) {
            throw std::runtime_error("Batch size cannot be larger than buffer size");
        }

        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
        for (int i = 0; i < batchSize; i++) {
            int number = _draw(startingIndex, _trainingBuffer->size() - 1);

            batch.push_back(_trainingBuffer->at(number));
        }
    } else {
        if (batchSize > _trainingBuffer->size()) {
            throw std::runtime_error("Batch size cannot be larger than buffer size");
        }

        std::unique_lock<std::mutex> lck(_lock);

        for (int i = 0; i < batchSize; i++) {
            int number = _draw(startingIndex, _trainingBuffer->size() - 1);

            batch.push_back(_trainingBuffer->at(number));
        }

        lck.unlock();	
    }
    
    return batch;
}


void ReplayBuffer::add(Utility::TD data) {
    if (_multiprocess) {
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
        if (_trainingBuffer->size() == _maxBufferSize) {
            _trainingBuffer->erase(_trainingBuffer->begin());	
            _trainingBuffer->push_back(data);
        }
        else {
            _trainingBuffer->push_back(data);
        }
    } else {
        std::unique_lock<std::mutex> lck(_lock);
        if (_trainingBuffer->size() == _maxBufferSize) {
            _trainingBuffer->erase(_trainingBuffer->begin());	
            _trainingBuffer->push_back(data);
        }
        else {
            _trainingBuffer->push_back(data);
        }
        lck.unlock();
    }
    
}

int ReplayBuffer::size() {
    if (_multiprocess) { 
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
        return _trainingBuffer->size();
    } else {   
        std::unique_lock<std::mutex> lck(_lock);
        int size = _trainingBuffer->size();
        lck.unlock();
        return size;
    }
}

void ReplayBuffer::clear() {
    if (_multiprocess) { 
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
        _trainingBuffer->clear();
        _bufferIndex = -1;
    } else {
        std::unique_lock<std::mutex> lck(_lock);
        _trainingBuffer->clear();
        _bufferIndex = -1;
        lck.unlock();
    }
}