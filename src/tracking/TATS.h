// C libs
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <signal.h>
#include <sys/wait.h> 

// C++ libs
#include <string> 
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <algorithm>


// OpenCV imports
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/tracking.hpp"

// Boost imports
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/variate_generator.hpp>

// Local imports
#include "../detection/CascadeDetector.h"
#include "../detection/RCNNDetector.h"
#include "../detection/YoloDetector.h"
#include "../detection/ArucoDetector.h"
#include "../wire/Wire.h"
#include "../servo/PCA9685.h"
#include "../servo/ServoKit.h"
#include "../util/util.h"
#include "../util/config.h"
#include "../network/SACAgent.h"
#include "../network/ReplayBuffer.h"
#include "../env/Env.h"
#include "../pid/PID.h"

// Pytorch imports
#include <torch/torch.h>

enum EVENT { 
    TARGET_DETECT,
    TARGET_LOST,
    LOCATION_UPDATE
};


class TATS {
private:

    // Other functions
    void __logRecords(Utility::TD trainData, int servo, bool reset);
    void __printOutput(Utility::TD trainData, int servo);

    pthread_cond_t __trainCond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t __trainLock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t __sleepLock = PTHREAD_MUTEX_INITIALIZER;

    Utility::Config  __config;
    SACAgent* __pidAutoTunerParent = nullptr;
    SACAgent* __pidAutoTunerChild = nullptr;
    Env* __servos = nullptr;
    pid_t __pid = -1;
    bool __parentMode;
    bool __initialized;

    // TODO:: Make numParams a dynamic calculation
    // Create a shared memory buffer for experiance replay
    int __numParams = 558298; // size of policy network! Will change if anything is edited in defaults
    boost::interprocess::managed_shared_memory* __segment;
    Utility::SharedBuffer* __sharedTrainingBuffer;
    Utility::sharedString* __s;
    ReplayBuffer* __replayBuffer;

    // Setup stats dirs and remove old logs
    std::string __losspath = "/trainingLoss.txt";
    std::string __avePath = "/episodeAverages.txt";
    std::string __stepPath = "/episodeStepRewards.txt";
    std::string __statDir = "/stat/";
    std::string __path;
    std::string __statPath;

    // Variables set by update function
    int __frameCount, __recheckFrequency, __trackerType;
    bool __useTracking, __draw, __showVideo;
    std::chrono::time_point<std::chrono::high_resolution_clock> __execbegin = std::chrono::high_resolution_clock::now();

    // Target location variables
    bool __programStart, __rechecked, __isTracking, __isSearching;
    int __lossCount, __trackCount, __searchCount, __searchCountMax, __lossCountMax;
    std::vector<std::string> __targets;
    std::string __currentTarget;
    cv::Rect2i __roi;
    cv::Rect2d __boundingBox;
    cv::Ptr<cv::Tracker> __tracker;

    // Object center coordinates
    int __frameCenterX;
    int __frameCenterY;

    // Object coordinates
    int __objX;
    int __objY;

    // Network variables
    Detect::ObjectDetector* __detector;
    std::string __weights;
    std::vector<std::string> __class_names;
    std::vector<struct Detect::DetectionData> __results;
    int __numInput;
    int __numHidden;
    int __numActions;
    int __actionHigh; 
    int __actionLow;
    
    // Program loop variants
    unsigned int __totalSteps = 0;
    bool __recentlyReset = false;
    int __doneCount = 0;

    // Train records
    int __numEpisodes[NUM_SERVOS] = { 0 };
    double __emaEpisodeRewardSum[NUM_SERVOS] = { 0.0 };
    double __stepAverageRewards[NUM_SERVOS] = { 0.0 };
    double __emaEpisodeStepSum[NUM_SERVOS] = { 0.0 };
    double __totalEpisodeRewards[NUM_SERVOS] = { 0.0 };
    double __totalEpisodeSteps[NUM_SERVOS] = { 0.0 };

    double __totalEpisodeObjPredError[NUM_SERVOS] = { 0.0 };
    double __stepAverageObjPredError[NUM_SERVOS] = { 0.0 };
    double __emaEpisodeObjPredErrorSum[NUM_SERVOS] = { 0.0 };
    
    // Training data
    double __predictedActions[NUM_SERVOS][NUM_ACTIONS] = { 0.0 };
    double __stateArray[NUM_INPUT] = { 0.0 };
    Utility::SD __currentState[NUM_SERVOS] = {{}};
    Utility::TD __trainData[NUM_SERVOS] = {{}};
    Utility::ED __eventData[NUM_SERVOS] = {{}};
    Utility::RD __resetResults = {};
    Utility::SR __stepResults = {};

    // Training variables
    bool __multiProcess;
    bool __trainMode;
    bool __isTraining;
    bool __initialRandomActions;
    int __numInitialRandomActions;
    bool __stepsWithPretrainedModel;
    int __numTransferLearningSteps;
    int __maxTrainingSteps;
    int __additionalDelay;
    int __batchSize;
    long __currentSteps;
    int __minBufferSize;
    int __maxBufferSize;
    int __numUpdates;
    double __rate;
    bool __variableFPS;
    int __FPSVariance;
    bool __varyResetAngles;
    double __resetAngleVariance;
    bool __useCurrentAngleForReset;
    int __updateRate;
    bool __useAutoTuning;
    int __maxStepsPerEpisode;
    int __resetAfterNInnactiveFrames;

    // Annealed ERE (Emphasizing Recent Experience)variables
    // https://arxiv.org/pdf/1906.04009.pdf
    double __N0;
    double __NT;
    double __T;
    double __t_i;

    // EMA Statistics
    double __percentage;
    double __timePeriods;
    double __emaWeight;

    // Other config variables
    bool __logOutput;
    int __numberServos;

    // Thread functions
    void __panTiltThread();
    void __autoTuneThread();

    // Thread variables
    std::thread* __panTiltT;
    std::thread* __autoTuneT;

public:
    TATS(Utility::Config config);
    ~TATS();
    void init(int pid);
    void update(cv::Mat& frame);
    void registerEventCallback();
    void syncTATS();
    bool trainTATSChildProcess();
};
