#include "ReplayBuffer.h"
#include <vector>
#include <random>
#include <iterator>

#include <cstdlib>
#include <ctime>
#include <iostream>


#include "../util/util.h"
#include "../util/data.h"


ReplayBuffer::ReplayBuffer(int maxBufferSize, Utility::SharedBuffer* buffer, bool multiprocess)
{
	_trainingBuffer = buffer;
	_mutex = new boost::interprocess::named_mutex(boost::interprocess::open_or_create, "ReplayBufferMutex");
	_maxBufferSize = maxBufferSize;
	_multiprocess = multiprocess;
	_trainBufferLock = PTHREAD_MUTEX_INITIALIZER;
}

ReplayBuffer::~ReplayBuffer() {
	if (_multiprocess) {
		boost::interprocess::named_mutex::remove("ReplayBufferMutex");
		delete _mutex;
	}
}

int ReplayBuffer::_draw(int min, int max)
{
	return std::uniform_int_distribution<int>{min, max}(eng);
}

Utility::TrainBuffer ReplayBuffer::ere_sample(int batchSize, int startingIndex)
{
	if (_multiprocess) {
		Utility::TrainBuffer batch;

		if (batchSize > _trainingBuffer->size()) {
			throw std::runtime_error("Batch size cannot be larger than buffer size");
		}

		boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
		for (int i = 0; i < batchSize; i++) {
			int number = _draw(startingIndex, _trainingBuffer->size() - 1);

			batch.push_back(_trainingBuffer->at(number));
		}
		
		return batch;
	} else {
		TrainBuffer batch;

		if (batchSize > _trainingBuffer->size()) {
			throw std::runtime_error("Batch size cannot be larger than buffer size");
		}

		if (pthread_mutex_lock(&_trainBufferLock) == 0) {

			for (int i = 0; i < batchSize; i++) {
				int number = _draw(startingIndex, _trainingBuffer->size() - 1);

				batch.push_back(_trainingBuffer->at(number));
			}
		}

		pthread_mutex_unlock(&_trainBufferLock);
		return batch;
	}
	
}


void ReplayBuffer::add(Utility::TD data)
{
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
		if (pthread_mutex_lock(&_trainBufferLock) == 0) {
			if (_trainingBuffer->size() == _maxBufferSize) {
				_trainingBuffer->erase(_trainingBuffer->begin());	
				_trainingBuffer->push_back(data);
			}
			else {
				_trainingBuffer->push_back(data);
			}
		}

		pthread_mutex_unlock(&_trainBufferLock);
	}
	
}

int ReplayBuffer::size() {
	if (_multiprocess) { 
		boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
		return _trainingBuffer->size();
	} else {
		
		if (pthread_mutex_lock(&_trainBufferLock) == 0) {
			return _trainingBuffer->size();
		}

		pthread_mutex_unlock(&_trainBufferLock);
	}
	
}

void ReplayBuffer::clear() {
	if (_multiprocess) { 
		boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
		_trainingBuffer->clear();
		_bufferIndex = -1;
	} else {
		if (pthread_mutex_lock(&_trainBufferLock) == 0) {
			_trainingBuffer->clear();
		}
		_bufferIndex = -1;
		
		pthread_mutex_unlock(&_trainBufferLock);
	}
}