#include "TATS.h"

namespace control {
    TATS::~TATS() {
        __stopFlag = true;
        __panTiltT->join();
        __autoTuneT->join();
        __eventCallT->join();
        delete __panTiltT;
        delete __autoTuneT;
        delete __eventCallT;
    }

    TATS::TATS(Utility::Config* config, control::ServoKit* servos) {
        using namespace Utility;
        
        // Misc variables
        __config = config;
        __numberServos = 2;
        __stopFlag = false;
        __callbacksRegistered = false;

        // Set up stats paths and files
        __path = get_current_dir_name();
        __statPath = __path + __statDir;
        mkdir(__statPath.c_str(), 0755);
        std::remove(( __statPath + __losspath ).c_str());

        for (int servo = 0; servo < __numberServos; servo++) {
            std::string servoPath = __statPath + std::to_string(servo);
            mkdir(servoPath.c_str(), 0755);
            
            std::remove(( servoPath + __avePath ).c_str());
            std::remove(( servoPath + __stepPath ).c_str());
        }

        // Initialize buffers in shared memory space
        if (__config->trainMode) {
            boost::interprocess::shared_memory_object::remove("SharedMemorySegment");
            __segment = boost::interprocess::managed_shared_memory(boost::interprocess::create_only, "SharedMemorySegment", (sizeof(Utility::TD) * __config->maxBufferSize + 1) + (sizeof(char) * __numParams + 1));
            const ShmemAllocator alloc_inst(__segment.get_segment_manager());
            __segment.construct<SharedBuffer>("SharedBuffer")(alloc_inst);
            __segment.construct<Utility::sharedString>("SharedString")("", alloc_inst);
        } 

        // Initialize other variables
        __parentMode = false;
        __initialized = false;
        __programStart = true;
        __logOutput = __config->logOutput;
        __frameCount = 0;
        __recheckFrequency = __config->recheckFrequency;
        __trackerType = __config->trackerType;
        __useTracking = __config->useTracking;
        __draw = __config->draw;
        
        // Target location variables
        __rechecked = false;
        __isTracking = false;
        __isSearching = false;
        __lossCount = 0;
        __trackCount = 0;
        __searchCount = 0;
        __searchCountMax = 5;
        __lossCountMax = __config->lossCountMax;
        __targets = __config->targets;
        __currentTarget = ""; 
        __targetId = -1;

        // Training variables
        __trainMode = __config->trainMode;
        __isTraining = false;
        __initialRandomActions = __config->initialRandomActions;
        __numInitialRandomActions = __config->numInitialRandomActions;
        __stepsWithPretrainedModel = __config->stepsWithPretrainedModel;
        __numTransferLearningSteps = __config->numTransferLearningSteps;  
        __batchSize = __config->batchSize;
        __maxTrainingSteps = __config->maxTrainingSteps;
        __minBufferSize = __config->minBufferSize;
        __maxBufferSize = __config->maxBufferSize;
        __numUpdates = __config->numUpdates;
        __rate = __config->trainRate;
        __multiProcess  = __config->multiProcess;
        __variableFPS = __config->variableFPS;
        __FPSVariance = __config->FPSVariance;
        __varyResetAngles = __config->varyResetAngles;
        __additionalDelay = 0;
        __resetAngleVariance = __config->resetAngleVariance;
        __useCurrentAngleForReset = __config->useCurrentAngleForReset;
        __updateRate = __config->updateRate;
        __useAutoTuning = __config->useAutoTuning;
        __maxStepsPerEpisode = __config->maxStepsPerEpisode;
        __resetAfterNInnactiveFrames = __config->resetAfterNInnactiveFrames;
        __currentReplayBuffSize = 0;

        // ERE Sampling variables
        __N0 = 0.996;
        __NT = 1.0;
        __T = __config->maxTrainingSteps;
        __t_i = 0;
        __currentSteps = 0;

        // EMA of steps and rewards (With 30% weight to new episodes; or 5 episode averaging)
        __percentage = (1.0 / 3.0);
        __timePeriods = (2.0 / __percentage) - 1.0;
        __emaWeight = (2.0 / (__timePeriods + 1.0));

        // Network variables
        __numHidden = __config->numHidden;
        __numActions = __config->numActions; 
        __actionHigh = __config->actionHigh; 
        __actionLow = __config->actionLow;
        __numInput =__config->numInput;
        __servos = new Env(servos, config);
        __actionType = __config->actionType;
    
        for (int servo = 0; servo < __numberServos; servo++) {
            __currentState[servo].actionType = __actionType;
            __trainData[servo].setActionType(__actionType);
        }
        
        __resetResults.setActionType(__actionType);
        __stepResults.setActionType(__actionType);
    }

    void TATS::registerCallback(EVENT eventType, std::function<void(control::INFO const&)> callback) {
        __callbacksRegistered = true;
        if (!__parentMode) {
            throw std::runtime_error("Can only register callbacks from an instance of TATS in a parent process");
        }

        if (__initialized) {
            throw std::runtime_error("Can only register callbacks before call to init()");
        }
        __callbacksRegistered = true;
        eventCallbacks[static_cast<int>(eventType)] = callback;
    }

    void TATS::init(int pid) {

        using namespace Utility;

        __pid = pid;

        // Parent process
        if (pid > 0 && !__initialized) {

            __pidAutoTuner = new SACAgent(
                __numInput, 
                __numHidden, 
                __numActions, 
                __actionHigh, 
                __actionLow
            );

            __parentMode = true;
            __initialized = true;

            if (__trainMode) {
                // Retrieve model params array from shared memory 
                __sharedTrainingBuffer = __segment.find<SharedBuffer>("SharedBuffer").first;

                if (__sharedTrainingBuffer == 0x0 || __maxBufferSize > __sharedTrainingBuffer->max_size()) {
                    throw std::runtime_error("could not construct shared buffer in memory");
                }

                __replayBuffer = new ReplayBuffer(__maxBufferSize, __sharedTrainingBuffer, __multiProcess);
                __s = __segment.find_or_construct<Utility::sharedString>("SharedString")("", __segment.get_segment_manager());
            }

            __execbegin = std::chrono::high_resolution_clock::now();
        
            if (__useTracking) {
                __tracker = createOpenCVTracker(__trackerType);
            }   

            // Load detector
            __weights = __path + __config->detectorPath;
            __class_names = __config->classes;

            switch(__config->detector) {
                case Utility::DetectorType::CASCADE:
                    __detector = new Detect::CascadeDetector(__weights);
                    break; 
                case Utility::DetectorType::ARUCO:
                    __detector = new Detect::ArucoDetector();
                    break; 
                case Utility::DetectorType::YOLO:
                    __detector = new Detect::YoloDetector(__weights, __class_names);
                    break; 
                default : 
                    __detector = new Detect::CascadeDetector(__weights);
            }

            if (!__trainMode) {
                __pidAutoTuner->eval();
            }

            __panTiltT = new std::thread(&TATS::__panTiltThread, this);
            // __eventCallT = new std::thread(&TATS::__eventCallbackThread, this);

            if (!__multiProcess && __trainMode) {
                __autoTuneT = new std::thread(&TATS::__autoTuneThread, this);
            }
            
        } else {

            __initialized = true;

            if (__trainMode) {
                __isTraining = true;
                
                // Retrieve the training buffer from shared memory
                __segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
                __sharedTrainingBuffer = __segment.find<SharedBuffer>("SharedBuffer").first;
                __replayBuffer = new ReplayBuffer(__config->maxBufferSize, __sharedTrainingBuffer, __multiProcess);
                __s = __segment.find_or_construct<sharedString>("SharedString")("", __segment.get_segment_manager());

                // Init a seperate instance of SAC for child process               
                __pidAutoTuner = new SACAgent(
                    __numInput, 
                    __numHidden, 
                    __numActions, 
                    __actionHigh, 
                    __actionLow
                );
            }
        }
    }

    void TATS::update(cv::Mat& frame) {

        // std::unique_lock<std::mutex> lck(__eventLock);

        if (!__parentMode) {
            throw std::runtime_error("Can only run update from an instance of TATS in a parent process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }
        
        std::chrono::steady_clock::time_point detectLoopStart = std::chrono::steady_clock::now(); // For delay
        
        // Vary FPS during training. 
        if (__trainMode && __variableFPS) {
            pthread_mutex_lock(&__sleepLock);
            int sleepTime = 0;
            sleepTime = __additionalDelay;
            pthread_mutex_unlock(&__sleepLock);
            
            if (sleepTime > 0) {
                // Note: Anything less than 10 milliseconds doesn't register well with std::chrono::steady_clock
                Utility::msleep(sleepTime);
            }
        }

        if (__isSearching) {
            __searchCount += 1;
            __queueEvent(EVENT::ON_SEARCH);
            onTargetUpdate(INFO(
                -1.0, 
                -1.0, 
                __targetId,
                !__isSearching
            ), EVENT::ON_SEARCH);
        }

        try {
            try {
                int cropSize = __config->dims[0];
                int offsetW = (frame.cols - cropSize) / 2;
                int offsetH = (frame.rows - cropSize) / 2;
                cv::Rect region(offsetW, offsetH, cropSize, cropSize);
                frame = frame(region).clone();
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                throw std::runtime_error("could not get image from camera");
            }

            if (!__useTracking) {
                goto detect;
            }

            if (__isTracking) {
                
                __isSearching = false;

                // Get the new tracking result
                if (!__tracker->update(frame, __boundingBox)) {
            lostTracking:
                    __isTracking = false;
                    __frameCount = 0.0;
                    __lossCount++;
                    __trackCount = 0;
                    __currentTarget = "";
                    __targetId = -1;
                    goto detect;
                }

                __roi = __boundingBox;

                // Chance to revalidate object tracking quality
                if (__frameCount >= __recheckFrequency) {
                    __frameCount = 0;
                    __rechecked = true;
                    goto detect;
                } else {
                    __frameCount++;
                }

            validated:
                // Determine object and frame centers
                __frameCenterX = frame.cols / 2;
                __frameCenterY = frame.rows / 2;
                __objX = __roi.x + (__roi.width / 2);
                __objY = __roi.y + (__roi.height / 2);

                // Enter State data
                auto wall_now = std::chrono::high_resolution_clock::now();
                double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - __execbegin).count() * 1e-9;
                
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                double spf = std::chrono::duration<double>(now - detectLoopStart).count();
                
                Utility::ED eventDataArray[2] {
                    {
                        false,
                        static_cast<double>(__frameCenterY - __objY),
                        static_cast<double>(__frameCenterY),
                        static_cast<double>(__roi.height),
                        static_cast<double>(__roi.y),
                        static_cast<double>(__objY),
                        timestamp,
                        spf
                    },
                    {
                        false,
                        static_cast<double>(__frameCenterX - __objX),
                        static_cast<double>(__frameCenterX),
                        static_cast<double>(__roi.width),
                        static_cast<double>(__roi.x),
                        static_cast<double>(__objX),
                        timestamp,
                        spf
                    }
                };

                try {
                    __queueEvent(EVENT::ON_UPDATE);
                    __servos->update(eventDataArray);
                    onTargetUpdate(INFO(
                        -1.0, 
                        -1.0, 
                        __targetId,
                        !__isSearching
                    ), EVENT::ON_UPDATE);
                } catch (...) {
                    throw std::runtime_error("cannot update event data");
                }

                // draw to frame
                if (__draw) {
                    Utility::drawPred(
                        1.0, 
                        __roi.x, 
                        __roi.y, 
                        __roi.x + __roi.width, 
                        __roi.y + __roi.height, 
                        frame
                    );
                }
            }
            else {

            detect:

                try {
                    std::vector<struct Detect::DetectionData> out = __detector->detect(frame, __draw, 1);
                    __results = out;
                } catch (const std::exception& e)
                {
                    std::cerr << e.what() << std::endl;
                    throw std::runtime_error("Could not detect target from frame");
                }
                
                Detect::DetectionData result;
                result.found = false;

                if (!__results.empty()) { 

                    // First time detect
                    if (__programStart || __isSearching) {
                        for (auto res : __results) {
                            
                            // Pick the first one we see
                            if (std::find(__targets.begin(), __targets.end(), res.target) != __targets.end()) {
                                result = res;
                                __roi = result.boundingBox;
                                __currentTarget = result.target;
                                __targetId = result.label;
                                break;
                            }
                        }         
                    } else if (__useTracking && __rechecked) {
                        __rechecked = false;
                        for (auto res : __results) {
                            result = res;
                            
                            // TODO::Determine criteria by which OpenCV Tracking is still accurate
                            // If they intersect
                            // if (((result.boundingBox & roi).area() > 0.0)) {
                            goto validated; 
                            // }  
                        }
                        
                        goto lostTracking;
                    } else {                
                        if (__results.size() > 1) {
                            
                            int bestDistance = __config->dims[0];
                            for (auto res : __results) {
                                // TODO:: Assumption below potentially wrong, use POT in the future
                                if (__currentTarget == res.target) {   
                                    cv::Point2i a = res.center;
                                    cv::Point2i b = (__roi.tl() + __roi.br()) / 2;
                                    int distance = std::sqrt<int>( (a.x-b.x) * (a.x-b.x) + (a.y-b.y) * (a.y-b.y) );

                                    if (distance <= bestDistance) {
                                        bestDistance = distance;
                                        result = res;
                                        __roi = result.boundingBox;
                                    }
                                }
                            } 
                        } else {
                            if (__currentTarget == __results[0].target) {
                                // TODO:: Trigger target detect event
                                result = __results[0];
                                __roi = result.boundingBox;
                            } 
                        }
                    }
                } 
                
                // If we confidently found the object we are looking for
                if (result.found) {
                    
                    // Update loop variants
                    __programStart = false;
                    __lossCount = 0;
                    __trackCount = (__trackCount + 1) % CHAR_MAX; 
                    __isSearching = false;

                    // Determine object and frame centers
                    __frameCenterX = frame.cols / 2;
                    __frameCenterY = frame.rows / 2;
                    __objX = result.center.x;
                    __objY = result.center.y;
                    
                    // Fill out state data
                    auto wall_now = std::chrono::high_resolution_clock::now();
                    double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - __execbegin).count() * 1e-9;
                    
                    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                    std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                    double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                    Utility::ED eventDataArray[2] {
                        {
                            false,
                            static_cast<double>(__frameCenterY - __objY),
                            static_cast<double>(__frameCenterY),
                            static_cast<double>(__roi.height),
                            static_cast<double>(__roi.y),
                            static_cast<double>(__objY),
                            timestamp,
                            spf
                        },
                        {
                            false,
                            static_cast<double>(__frameCenterX - __objX),
                            static_cast<double>(__frameCenterX),
                            static_cast<double>(__roi.width),
                            static_cast<double>(__roi.x),
                            static_cast<double>(__objX),
                            timestamp,
                            spf
                        }
                    };

                    try {
                        __queueEvent(EVENT::ON_UPDATE);
                        onTargetUpdate(INFO(
                            -1.0, 
                            -1.0, 
                            __targetId,
                            !__isSearching
                        ), EVENT::ON_UPDATE);
                        __servos->update(eventDataArray);
                    } catch (...) {
                        throw std::runtime_error("cannot update event data");
                    }

                    if (__useTracking && !__isTracking) {

                        try {
                            __tracker = Utility::createOpenCVTracker(__trackerType);
                            if (__tracker->init(frame, __roi)) {
                                __isTracking = true;
                            } 
                        } catch (const std::exception& e) {
                            std::cerr << e.what() << std::endl;
                            throw std::runtime_error("Could not init opencv tracker");
                        }
                    }
                }
                else if (!__config->trainMode && __config->usePOT) {
                    __lossCount++;
                    __frameCount = 0.0;
                    __rechecked = false;

                    // Target is out of sight, inform PID's, model, and servos
                    if (__lossCount >= __lossCountMax) {
                        // TODO:: Tigger on target lost track
                        __trackCount = 0;
                        __isSearching = true;
                        __isTracking = false;
                        __lossCount = 0;

                        // Enter state data
                        __frameCenterX = frame.cols / 2;
                        __frameCenterY = frame.rows / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - __execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        Utility::ED eventDataArray[2] {
                            {
                                true,
                                static_cast<double>(__frameCenterY),
                                static_cast<double>(__frameCenterY),
                                0.0,
                                0.0,
                                static_cast<double>(__frameCenterY * 2), // Max error
                                timestamp,
                                spf
                            },
                            {
                                true,
                                static_cast<double>(__frameCenterX),
                                static_cast<double>(__frameCenterX),
                                0.0,
                                0.0,
                                static_cast<double>(__frameCenterX * 2),
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            __queueEvent(EVENT::ON_LOST);
                            onTargetUpdate(INFO(
                                -1.0, 
                                -1.0, 
                                __targetId,
                                !__isSearching
                            ), EVENT::ON_LOST);
                            __servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    } else if (!__isSearching && !__programStart && __trackCount > 3) {
                        
                        // Get the prediction location of object relative to frame center
                        if (__config->logOutput) {
                            std::cout << "Using Predictive Object Tracking: frame # " << std::to_string(__lossCount) << std::endl;
                        }
                        
                        // Enter state data
                        double locations[__numberServos] = { 0.0 };
                        __servos->getPredictedObjectLocation(locations);
                        __frameCenterX = __config->dims[1] / 2;
                        __frameCenterY = __config->dims[0] / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - __execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        Utility::ED eventDataArray[2] {
                            {
                                false,
                                static_cast<double>(__frameCenterY) - locations[0],
                                static_cast<double>(__frameCenterY),
                                0.0,
                                0.0,
                                locations[0], // Predicted location
                                timestamp,
                                spf
                            },
                            {
                                false,
                                static_cast<double>(__frameCenterX) - locations[1],
                                static_cast<double>(__frameCenterX),
                                0.0,
                                0.0,
                                locations[1],
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            __queueEvent(EVENT::ON_UPDATE);
                            onTargetUpdate(INFO(
                                -1.0, 
                                -1.0, 
                                __targetId,
                                !__isSearching
                            ), EVENT::ON_UPDATE);
                            __servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    }
                }
                else {
                    __lossCount++;
                    __rechecked = false;

                    // Target is out of sight, inform PID's, model, and servos
                    if (__lossCount >= __lossCountMax) {
                        
                        __isSearching = true;
                        __isTracking = false;
                        __lossCount = 0;
                        __trackCount = 0;

                        // Enter State data
                        __frameCenterX = frame.cols / 2;
                        __frameCenterY = frame.rows / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - __execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        Utility::ED eventDataArray[2] {
                            {
                                true,
                                static_cast<double>(frame.rows), // Obj not on screen, Max error
                                static_cast<double>(__frameCenterY),
                                0.0,
                                0.0,
                                static_cast<double>(__frameCenterY * 2), 
                                timestamp,
                                spf
                            },
                            {
                                true,
                                static_cast<double>(frame.cols),
                                static_cast<double>(__frameCenterX),
                                0.0,
                                0.0,
                                static_cast<double>(__frameCenterX * 2),
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            __queueEvent(EVENT::ON_LOST);
                            onTargetUpdate(INFO(
                                -1.0, 
                                -1.0, 
                                __targetId,
                                !__isSearching
                            ), EVENT::ON_LOST);
                            __servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    } 
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Issue detecting target from image");
        }

        // lck.unlock();
    }

    void TATS::__panTiltThread() {

        if (!__parentMode) {
            throw std::runtime_error("panTiltThread can only run from a parent process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }

        using namespace Utility;
        std::mt19937 eng{ std::random_device{}() };
        torch::Device device(torch::kCPU);
        std::default_random_engine _generator;
        std::normal_distribution<double> _distribution;

        auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
        double rate = static_cast<double>(__updateRate);

        try {
            __resetResults = __servos->reset();	
        } catch (...) {
            throw std::runtime_error("cannot reset servos");
        }
        
        for (int servo = 0; servo < __numberServos; servo++) {
            __currentState[servo] = __resetResults.servos[servo];
        }

        while (!__stopFlag) {
            bool overriden = false;
            if (__useAutoTuning) {

                if (!__servos->isDone()) {
                    __recentlyReset = false;
                    __doneCount = 0;

                    for (int i = 0; i < __numberServos; i++) {

                        // Query network and get PID gains
                        if (__trainMode) {
                            if (__initialRandomActions && __numInitialRandomActions > 0) {
                                for (int a = 0; a < __numActions; a++) {
                                    __predictedActions[i][a] = std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                                }
                            } else if (__stepsWithPretrainedModel && __numTransferLearningSteps > 0) {
                                __currentState[i].getStateArray(__stateArray);
                                at::Tensor actions = __pidAutoTuner->get_action(torch::from_blob(__stateArray, { 1, __numInput }, options), true);
                                actions.to(torch::kCPU);
                                if (__numActions > 1) {
                                    auto actions_a = actions.accessor<double, 1>();
                                    for (int a = 0; a < __numActions; a++) {
                                        __predictedActions[i][a] = actions_a[a];
                                    }
                                }
                                else {
                                    __predictedActions[i][0] = actions.item().toDouble();
                                }
                            }
                            else {

                                if (__initialRandomActions) {
                                    __initialRandomActions = false;
                                    std::cout << "WARNING: Done generating initial random actions" << std::endl;
                                } else if (__stepsWithPretrainedModel) {
                                    __stepsWithPretrainedModel = false;
                                    std::cout << "WARNING: Done taking steps with pre-trained model in evaluation mode" << std::endl;
                                }
                                
                                // Perform Inference, get action(s) 
                                __currentState[i].getStateArray(__stateArray);
                                at::Tensor actions = __pidAutoTuner->get_action(torch::from_blob(__stateArray, { 1, __numInput }, options), true);
                                actions.to(torch::kCPU);
                                if (__numActions > 1) {
                                    auto actions_a = actions.accessor<double, 1>();
                                    for (int a = 0; a < __numActions; a++) {
                                        __predictedActions[i][a] = actions_a[a];
                                    }
                                }
                                else {
                                    __predictedActions[i][0] = actions.item().toDouble();
                                }                         
                            }
                        }
                        else {
                            __currentState[i].getStateArray(__stateArray);
                            at::Tensor actions = __pidAutoTuner->get_action(torch::from_blob(__stateArray, { 1, __numInput }, options), false);
                            actions.to(torch::kCPU);
                            if (__numActions > 1) {
                                auto actions_a = actions.accessor<double, 1>();
                                for (int a = 0; a < __numActions; a++) {
                                    __predictedActions[i][a] = actions_a[a];
                                }
                            }
                            else {
                                __predictedActions[i][0] = actions.item().toDouble();
                            }
                        }
                    }

                    try {
                        overriden = actionOverride(__predictedActions, __actionType);
                        __stepResults = __servos->step(__predictedActions, true, rate);		
                        __totalSteps = (__totalSteps + 1) % INT_MAX;
                        onServoUpdate(
                            __stepResults.servos[1].nextState.currentAngle, 
                            __stepResults.servos[0].nextState.currentAngle
                        );
                    } catch(const std::exception& e) {
                        std::cerr << e.what() << std::endl;
                        throw std::runtime_error("cannot step with servos");
                    }
                    
                    bool reset = false;
                    bool updated = false;

                    // Update loop varients
                    for (int servo = 0; servo < __numberServos; servo++) {

                        // Next state becomes the current state next time we step
                        __trainData[servo] = __stepResults.servos[servo];
                        __trainData[servo].currentState = __currentState[servo];
                        __currentState[servo] = __trainData[servo].nextState;

                        // If servo is disabled, null record
                        if (__trainData[servo].empty) {
                            continue;
                        }
                        
                        if (__trainMode) {

                            if (__initialRandomActions && !updated) {
                                updated = true;
                                __numInitialRandomActions--;
                            } else if (__stepsWithPretrainedModel && !updated) {
                                updated = true;
                                __numTransferLearningSteps--;
                            }

                            // If early episode termination
                            if (__config->episodeEndCap) {
                                if (__totalEpisodeSteps[servo] > __maxStepsPerEpisode) {
                                    reset = true;
                                }
                            }

                            // Add to replay buffer for training
                            if (__replayBuffer->size() <= __maxBufferSize) {
                                __replayBuffer->add(__trainData[servo]);
                            }

                            // logging
                            __logRecords(__trainData[servo], servo, reset);
                        }

                        // Debug output
                        if (__logOutput) {
                            __printOutput(__trainData[servo], servo);
                        }    

                        // Start training thread/processes when ready
                        if (__trainMode) {
                            // Inform child process to start training
                            if (__multiProcess) {
                                if (!__isTraining && __replayBuffer->size() > (__minBufferSize)) {
                                    std::cout << "Sending train signal..." << std::endl;
                                    __isTraining = true;
                                    kill(this->__pid, SIGUSR1);
                                } 
                            } else if (!__isTraining && __replayBuffer->size() > (__minBufferSize)) {
                                // Inform autotune thread start training 
                                pthread_mutex_lock(&__trainLock);
                                pthread_cond_broadcast(&__trainCond);
                                pthread_mutex_unlock(&__trainLock);
                            }
                        }
                    }

                    if (reset) { goto reset; }   
                }
                else if (!__recentlyReset) {
                    __doneCount++;
                    __recentlyReset = true;

                    reset:

                    double overrides[NUM_SERVOS][NUM_ACTIONS] = { 0 };
                    overriden = actionOverride(overrides, Utility::ActionType::ANGLE); // We want to reset to an angle

                    // Vary env sync rate and FPS to simulate different latency configurations
                    if (!overriden && __trainMode) {
                        rate = static_cast<double>(__updateRate);
                        double adjustment = (( rate ) / 2.0) * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                        rate += adjustment;

                        // Locked to avoid race condition on additionalDelay with detectThread
                        pthread_mutex_lock(&__sleepLock);
                        if (__variableFPS) {
                            __additionalDelay = static_cast<int>(std::round(__FPSVariance * std::uniform_real_distribution<double>{ 0, 1.0 }(eng)));
                        } 
                        pthread_mutex_unlock(&__sleepLock);
                    }
                    
                    // Vary angle of reset after set amount of time to enable AI to work at any reset angle
                    if (__trainMode && __varyResetAngles) {
                        
                        double newAngles[NUM_SERVOS] = { 0.0 };
                        for (int servo = 0; servo < __numberServos; servo++) {

                            if (overriden) {
                                newAngles[servo] = Utility::rescaleAction(overrides[servo][0], __actionLow, __actionHigh);
                            } else {
                                // newAngles[servo] =  Utility::rescaleAction(std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng), __actionLow, __actionHigh);
                                newAngles[servo] =  __resetAngleVariance * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                            }
                        }
                        __resetResults = __servos->reset(newAngles);

                    } else { 
                        if (overriden) {
                            double newAngles[NUM_SERVOS] = { 0.0 };
                            for (int servo = 0; servo < __numberServos; servo++) {
                                newAngles[servo] = Utility::rescaleAction(overrides[servo][0], __actionLow, __actionHigh);
                            }
                            __resetResults = __servos->reset(newAngles);
                        
                        } else {
                            __resetResults = __servos->reset(__useCurrentAngleForReset);         
                        } 
                    }

                    onServoUpdate(
                        overriden ? overrides[1][0] : __resetResults.servos[1].currentAngle,
                        overriden ? overrides[0][0] : __resetResults.servos[0].currentAngle
                    );

                    // onServoUpdate(
                    //     __resetResults.servos[1].currentAngle,
                    //     __resetResults.servos[0].currentAngle
                    // );
                    
                    // Hold onto reset results
                    for (int servo = 0; servo < __numberServos; servo++) {
                        __currentState[servo] = __resetResults.servos[servo];
                    }   		
                } else {
                    __doneCount++;
                    double overrides[NUM_SERVOS][NUM_ACTIONS] = { 0 };
                    overriden = actionOverride(overrides, Utility::ActionType::ANGLE); // We want to reset to an angle and not a speed

                    if (__doneCount >= __resetAfterNInnactiveFrames && __resetAfterNInnactiveFrames > 0) {
                        if (__trainMode) {
                            double newAngles[NUM_SERVOS] = { 0.0 };
                            for (int servo = 0; servo < __numberServos; servo++) {

                                if (overriden) {
                                    newAngles[servo] = Utility::rescaleAction(overrides[servo][0], __actionLow, __actionHigh);
                                } else {
                                    // newAngles[servo] =  Utility::rescaleAction(std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng), __actionLow, __actionHigh);
                                    newAngles[servo] =  __resetAngleVariance * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                                }
                            }
                            __resetResults = __servos->reset(newAngles);
                        } else {
                            __resetResults = __servos->reset();            
                        }
                        
                        // onServoUpdate(
                        //     __resetResults.servos[1].currentAngle,
                        //     __resetResults.servos[0].currentAngle
                        // );

                        onServoUpdate(
                            overriden ? overrides[1][0] : __resetResults.servos[1].currentAngle,
                            overriden ? overrides[0][0] : __resetResults.servos[0].currentAngle
                        );
                        __doneCount = 0;
                        
                    } else {
                        if (overriden) {
                            double newAngles[NUM_SERVOS] = { 0.0 };
                            for (int servo = 0; servo < __numberServos; servo++) {
                                newAngles[servo] = Utility::rescaleAction(overrides[servo][0], __actionLow, __actionHigh);
                            }
                            __resetResults = __servos->reset(newAngles);
                        
                        } else {
                            __resetResults = __servos->reset(__useCurrentAngleForReset);            
                        } 

                        onServoUpdate(
                            overriden ? overrides[1][0] : __resetResults.servos[1].currentAngle,
                            overriden ? overrides[0][0] : __resetResults.servos[0].currentAngle
                        );
                    }
                    
                    // Hold onto reset results
                    for (int servo = 0; servo < __numberServos; servo++) {
                        __currentState[servo] = __resetResults.servos[servo];
                    }
                }    
            }
            else {
                if (!__servos->isDone()) {
                    for (int i = 0; i < __numberServos ; i++) {
                        __predictedActions[i][0] = __config->defaultGains[0];
                        __predictedActions[i][1] = __config->defaultGains[1];
                        __predictedActions[i][2] = __config->defaultGains[2];
                    }

                    try {
                        overriden = actionOverride(__predictedActions, __actionType);
                        __stepResults = __servos->step(__predictedActions, false);
                        onServoUpdate(
                            __stepResults.servos[1].nextState.currentAngle, 
                            __stepResults.servos[0].nextState.currentAngle
                        );
                        double state[NUM_INPUT];
                        for (int i = 0; i < __numberServos ; i++) {
                            
                            if (__stepResults.servos[i].empty) {
                                continue;
                            }
                        }
                        
                    } catch (...) {
                        throw std::runtime_error("cannot step with servos");
                    }
                }
                else {

                    try {
                        __resetResults = __servos->reset(__useCurrentAngleForReset);
                    } catch (...) {
                        throw std::runtime_error("cannot reset servos");
                    }
                }	
            }
        }
    }

    void TATS::__autoTuneThread() {

        if (!__parentMode) {
            throw std::runtime_error("autoTuneThread can only run from a parent process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }

        using namespace Utility;

        __replayBuffer->clear();


        // Wait for the buffer to fill
        pthread_mutex_lock(&__trainLock);
        while (__replayBuffer->size() <= __minBufferSize) {
            pthread_cond_wait(&__trainCond, &__trainLock);
        }
        pthread_mutex_unlock(&__trainLock);

        while (__isTraining && !__stopFlag) {

            double N = static_cast<double>(__replayBuffer->size());
            __t_i += 1;
            double n_i = __N0 + (__NT - __N0) * (__t_i / __T);
            int cmin = N - ( __minBufferSize );
            
            // Check if training is over
            if (__currentSteps >= __maxTrainingSteps) {
                __replayBuffer->clear();
                __isTraining = false;
                std::cout << "Training session over!!" << std::endl;
                return;
            }
            else {
                __currentSteps += 1;
            }

            for (int k = 0; k < __numUpdates; k += 1) {
                int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / __numUpdates)), cmin);

                TrainBuffer batch = __replayBuffer->ere_sample(__batchSize, startingRange);
                __pidAutoTuner->update(batch.size(), &batch);
                
            }

            long milis = static_cast<long>(1000.0 / __rate);
            Utility::msleep(milis);
        }
    }

    void TATS::__eventCallbackThread() {

        if (__callbacksRegistered) {
            return;
        }

        if (!__parentMode) {
            throw std::runtime_error("pingEnvThread can only run from a parent process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }

        while (!__stopFlag) {
            
            // Block until new update from ENV, then trigger newest event.
            // __servos->waitForUpdate();
            // std::unique_lock<std::mutex> lck(__eventLock); // this->update(cv::mat frame) holds lock otherwise

            // INFO event = __getINFO();

            // int id = __eventId;
            // lck.unlock();

            // __triggerCallback(EVENT(id), event);
        }
    }

    control::INFO TATS::__getINFO() {
        double angles[__numberServos];
        __servos->getCurrentAngle(angles);
        
        return INFO(
            angles[1], 
            angles[0], 
            __targetId,
            !__isSearching
        );
    }

    void TATS::syncTATS() {
        if (!__parentMode) {
            throw std::runtime_error("Can only run update from an instance of TATS in a parent process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }

        try {
            __pidAutoTuner->load_policy(__s);
        } catch (...) {
            throw std::runtime_error("Cannot refrech SAC network parameters");
        }
    }

    bool TATS::trainTATSChildProcess()  {

        using namespace Utility;

        if (__parentMode) {
            throw std::runtime_error("Can only train from an instance of TATS in a child process");
        }

        if (!__initialized) {
            throw std::runtime_error("init(pid) must be called be for other functions are executed");
        }

        // Small fix to prevent overtraining on stale data
        double N = static_cast<double>(__replayBuffer->size());
        if (__currentReplayBuffSize == N) {
            return false;
        }  else {
            __currentReplayBuffSize = N;
        }

        // Increment ERE training variables
        __t_i += 1;
        double n_i = __N0 + (__NT - __N0) * (__t_i / __T);
        int cmin = N - ( __minBufferSize );
            
        // Check if training is over
        if (__currentSteps >= __maxTrainingSteps) {
            __replayBuffer->clear();
            __isTraining = false;
            return false;
        }
        else {
            __currentSteps += 1;
            
            // Perform a training session
            for (int k = 0; k < __numUpdates; k += 1) {
                int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / __numUpdates)), cmin);
                Utility::TrainBuffer batch = __replayBuffer->ere_sample(__batchSize, startingRange);
                __pidAutoTuner->update(batch.size(), &batch);
            }

            // Write values to shared memory for parent to read
            try {
                __pidAutoTuner->save_policy(__s);
            } catch (...) {
                throw std::runtime_error("Cannot save policy params to shared memory array");
            }

            return true;
        }  
    }

    void TATS::__logRecords(Utility::TD trainData, int servo, bool reset) {

        if (trainData.done || reset) {       
            __numEpisodes[servo] += 1;
            __emaEpisodeRewardSum[servo] = (__totalEpisodeRewards[servo] - __emaEpisodeRewardSum[servo]) * __emaWeight + __emaEpisodeRewardSum[servo];
            __emaEpisodeStepSum[servo] = (__totalEpisodeSteps[servo] - __emaEpisodeStepSum[servo]) * __emaWeight + __emaEpisodeStepSum[servo];
            __emaEpisodeObjPredErrorSum[servo] = (__totalEpisodeObjPredError[servo] - __emaEpisodeObjPredErrorSum[servo]) * __emaWeight + __emaEpisodeObjPredErrorSum[servo];

            // Log Episode averages
            std::string episodeData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                                    (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                                    + std::to_string(__totalSteps) + ','
                                    + std::to_string(__numEpisodes[servo]) + ','
                                    + std::to_string(__totalEpisodeSteps[servo] > 0.0 ? __totalEpisodeRewards[servo] / __totalEpisodeSteps[servo] : 0.0) + ','
                                    + std::to_string(__totalEpisodeSteps[servo]) + ','
                                    + std::to_string(__emaEpisodeRewardSum[servo] / __emaEpisodeStepSum[servo]) + ','
                                    + std::to_string(__emaEpisodeStepSum[servo]) + ','
                                    + std::to_string(__emaEpisodeObjPredErrorSum[servo] / __emaEpisodeStepSum[servo]);
            
            Utility::appendLineToFile(__path + "/stat/" + std::to_string(servo) + "/episodeAverages.txt", episodeData);

            __totalEpisodeSteps[servo] = 0.0;
            __totalEpisodeRewards[servo] = 0.0;
            __totalEpisodeObjPredError[servo] = 0.0;
        } else {
            // Average reward in a step
            __totalEpisodeSteps[servo] += 1.0;						
            __totalEpisodeRewards[servo] += trainData.errors[0];
            __totalEpisodeObjPredError[servo] += trainData.errors[1];

            __stepAverageRewards[servo] = (trainData.errors[0] - __stepAverageRewards[servo]) * __emaWeight + __stepAverageRewards[servo];
            __stepAverageObjPredError[servo] = (trainData.errors[1] - __stepAverageObjPredError[servo]) * __emaWeight + __stepAverageObjPredError[servo];

            std::string stepData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                                (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                                + std::to_string(__totalSteps) + ','
                                + std::to_string(trainData.errors[0]) + ',' 
                                + std::to_string(__stepAverageRewards[servo])  + ','
                                + std::to_string(trainData.errors[1])  + ','
                                + std::to_string(__stepAverageObjPredError[servo]);

            Utility::appendLineToFile(__path + "/stat/" + std::to_string(servo) + "/episodeStepRewards.txt", stepData);
        }   
    }

    void TATS::__printOutput(Utility::TD trainData, int servo) {
        double state[NUM_INPUT];
        std::cout << (servo ? "Pan step info:" : "Tilt step info:") << std::endl;
        std::cout << "Next State: ";
        trainData.nextState.getStateArray(state);
        for (int j = 0; j < NUM_INPUT; j++) {
            std::cout << std::to_string(state[j]) << ", ";
        }
        std::cout << std::endl;

        std::cout << "Current State: ";
        trainData.currentState.getStateArray(state);
        for (int j = 0; j < NUM_INPUT; j++) {
            std::cout << std::to_string(state[j]) << ", ";
        }
        std::cout << std::endl;

        std::cout << "FPS Delay: ";
        std::cout << std::to_string(__additionalDelay);
        std::cout << std::endl;

        std::cout << "Reward: ";
        std::cout << std::to_string(trainData.reward);
        std::cout << std::endl;

        std::cout << "Done: ";
        std::cout << std::to_string(trainData.done);
        std::cout << std::endl;

        std::cout << "Actions: ";
        
        switch (__actionType)
        {
            case Utility::ActionType::PID:
                for (int j = 0; j < 3; j++) {
                    std::cout << std::to_string(Utility::rescaleAction(trainData.actions[j], __config->actionLow, __config->actionHigh)) << ", ";
                }  
                break;
            case Utility::ActionType::ANGLE:
                std::cout << std::to_string(Utility::rescaleAction(trainData.actions[0], __config->anglesLow[servo], __config->anglesHigh[servo])) << ", ";
                break;
            case Utility::ActionType::SPEED:
                std::cout << std::to_string(trainData.actions[0]) << ", ";
                break;
            default:
                throw std::runtime_error("Invalid action type provided");
        }
     
        if (__config->usePOT) {
            std::cout << std::to_string(Utility::rescaleAction(trainData.actions[1], 0.0, __config->dims[servo])) << std::endl;
        } else {
            std::cout << std::endl;
        }

        std::cout << "Errors: ";
        std::cout << std::to_string(trainData.errors[0]) << " "; 
        if (__config->usePOT) {
            std::cout << std::to_string(trainData.errors[1]);
        }
        std::cout << std::endl;
    };

    void TATS::__queueEvent(EVENT eventType) {
        __eventId = static_cast<int>(eventType);
    }

    void TATS::__triggerCallback(EVENT eventType, INFO event){
        int e = static_cast<int>(eventType);
        if(eventCallbacks[e]) {
            eventCallbacks[e](event);
        }
    }

    void TATS::onServoUpdate(double pan, double tilt) {

    }

    void TATS::onTargetUpdate(control::INFO info, EVENT eventType) {

    }

    bool TATS::actionOverride(double actions[NUM_SERVOS][NUM_ACTIONS], Utility::ActionType actionType) {
        return false;
    }

    bool TATS::isTraining() {
        return __isTraining;
    }

    double TATS::updateRate() {
        
    }
};