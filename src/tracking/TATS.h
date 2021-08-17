// C libs
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <signal.h>
#include <sys/wait.h> 
#include <mutex>

// C++ libs
#include <string> 
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <algorithm>
#include <future>         // std::async, std::future


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
#include "../servo/PCA9685.h"
#include "../util/util.h"
#include "../util/config.h"
#include "../network/SACAgent.h"
#include "../network/ReplayBuffer.h"
#include "../env/Env.h"
#include "../pid/PID.h"

// Pytorch imports
#include <torch/torch.h>

namespace control {

    enum class EVENT { 
        ON_DETECT, /* Target entity first encounter */
        ON_LOST,   /* Target entity lost */
        ON_UPDATE, /* Target entity locatioin update */
        ON_SEARCH,  /* Target Entity search routine */
        size
    };

    struct EVENT_INFO {
        double pan;
        double tilt;
        int id;
        bool tracking;

        EVENT_INFO(
            double pan, 
            double tilt, 
            int id, 
            bool t) : 
            pan(pan), 
            tilt(tilt), 
            id(id),  
            tracking(t) {}

        EVENT_INFO() : 
            pan(-1.0),
            tilt(-1.0), 
            id(0), 
            tracking(false)
        {}
    } typedef INFO;

    class TATS {

    private:

        // Other functions
        void __logRecords(Utility::TD trainData, int servo, bool reset);
        void __printOutput(Utility::TD trainData, int servo);
        
        /***
         * @brief Returns struct containing metadata on servo and target locations
         * @return INFO
         */
        control::INFO __getINFO();

        pthread_cond_t __trainCond = PTHREAD_COND_INITIALIZER;
        pthread_mutex_t __trainLock = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t __sleepLock = PTHREAD_MUTEX_INITIALIZER;

        // Misc class variables
        Utility::Config*  __config = nullptr;
        SACAgent* __pidAutoTuner = nullptr;
        Env* __servos = nullptr;
        pid_t __pid = -1;
        bool __parentMode;
        bool __initialized;
        bool __callbacksRegistered;

        // TODO:: Make numParams a dynamic calculation
        // Create a shared memory buffer for experiance replay
        int __numParams = 558298; // size of policy network! Will change if anything is edited in defaults
        // boost::interprocess::managed_shared_memory* __segment;
        Utility::SharedBuffer* __sharedTrainingBuffer;
        Utility::sharedString* __s;
        ReplayBuffer* __replayBuffer;
        boost::interprocess::managed_shared_memory __segment;

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
        int __targetId;
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
        double __previousAngles[NUM_SERVOS] = { 0.0 };
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
        int __currentReplayBuffSize;

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

        // Threads
        bool __stopFlag;
        void __panTiltThread();
        void __autoTuneThread();
        void __eventCallbackThread();
        std::thread* __panTiltT;
        std::thread* __autoTuneT;
        std::thread* __eventCallT;

        // Event callbacks
        std::array<std::function<void(control::INFO const&)>, static_cast<int>(EVENT::size)> eventCallbacks;
        void __triggerCallback(EVENT eventType, INFO event);
        void __queueEvent(EVENT eventType);
        std::mutex __eventLock;
        int __eventId;


    public:
        /***
         * @brief Constructor
         * @param config Configuration object for option configuration
         * @param servos Pointer to servos TATS will control
         */
        TATS(Utility::Config* config, control::ServoKit* servos);
        ~TATS();

        /***
         * @brief Initializes TATS to desired mode (Train vs Eval modes)
         * @param pid Set pid > 0 for TATS evaluation mode. If in train mode, provide pid provided from call to fork. 
         *        Should be 0 for a child process instance of TATS.
         *        For example:
         * 
         *        int pid = fork();
         *        if (pid > 0) {
         *           // Parent
         *           TATS.init(pid);
         *        } else {
         *           // Child
         *           TATS.init(pid);
         *        }
         *         
         * @return void
         */
        void init(int pid);

        /***
         * @brief Detects target from frame and moves servos towards it
         * @param frame Image to run  detection on
         * @return void
         */
        void update(cv::Mat& frame);

        /***
         * @brief After a refresh of the servos (will NOT be triggered on every this->update(cv::Mat& frame)), 
         *        calls specified function immediately event. Called asynchronously after servos update.
         * @param eventType Condition for callback
         * @param callback Function to be asynchronously processed
         * @return void
         */
        void registerCallback(EVENT eventType, std::function<void(control::INFO const&)> callback);

        /***
         * @brief Called once on every new update of TARGET. 
                  Always returns the newest target detections.
                  Simply override with desired behavior in main.cpp.
                  EXAMPLE: void onTargetUpdate(control::INFO const&, EVENT eventType) { ... custom code here ...; }
                  NOTE: Overrides prototype for all instances of TATS
         * @param info Returns only current target info. Angle info should be obtained from onServoUpdate.
         * @param eventType Condition triggered on this update
         * @return void
         */
        void onTargetUpdate(control::INFO info, EVENT eventType) __attribute__((weak));

        /***
         * @brief Called once on every new update of Servos. Async Call. 
                  Always returns the newest PAN/TILT angles.
                  Simply override with desired behavior in main.cpp.
                  EXAMPLE: void onServoUpdate(double pan, double tilt) { ... custom code here ...; }
                  NOTE: Overrides prototype for all instances of TATS
         * @param pan In angles
         * @param tilt In angles
         * @return void
         */
        void onServoUpdate(double pan, double tilt) __attribute__((weak));

        /***
         * @brief Called once before Servos.step(). Allows
                  the user to override AI actions. AI learns from these action instead.
                  Simply override with desired behavior in main.cpp.
                  EXAMPLE: double actionOverride(double actions[2])  { ... custom code here ...; }
                  NOTE: Overrides prototype for all instances of TATS
         * @param actions Actions to provide to the AI [-1.0, 1.0]
         * @return void
         */
        bool actionOverride(double actions[NUM_SERVOS][NUM_ACTIONS]) __attribute__((weak));
        
        /***
         * @brief For train mode only. From parent process, load new network params
         * @return void
         */
        void syncTATS();

        /***
         * @brief For train mode only. From child process, train on collected experiences
         * @return void
         */
        bool trainTATSChildProcess();

        /***
         * @brief Returns true if model is currently training
         * @return bool train status
         */
        bool isTraining();

        /***
         * @brief Returns update rate of the servos
         * @return double current servo update
         */
        double updateRate();
    };
};