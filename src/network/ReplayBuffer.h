#pragma once
#include <vector>
#include <random>
#include <iterator>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

#include "../util/util.h"
#include "../util/data.h"

#include <condition_variable>
#include <mutex>

class ReplayBuffer
{
private:
	Utility::SharedBuffer* _trainingBuffer;
	int _maxBufferSize;
	int _bufferIndex = -1;
	bool _multiprocess;

	// Create a random device and use it to generate a random seed
	std::mt19937 eng{ std::random_device{}() };
	std::mutex _lock;
	boost::interprocess::named_mutex* _mutex;

public:
	ReplayBuffer(int maxBufferSize, Utility::SharedBuffer* buffer, bool multiprocess = false);
	~ReplayBuffer();
	Utility::TrainBuffer ere_sample(int batchSize = 32, int startingIndex = 0);
	void add(Utility::TD data);
	int size();
	void clear();
	int _draw(int min, int max);
};

