// C libs
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/prctl.h>
#include <signal.h>
#include <sys/wait.h> 

// C++ libs
#include <string> 
#include <vector>
#include <iostream>
#include <chrono>
#include <csignal>
#include <random>
#include <ctime>
#include <thread>
#include <algorithm>

// Pytorch imports
#include <torch/torch.h>

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
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
#include "./src/detection/CascadeDetector.h"
#include "./src/detection/RCNNDetector.h"
#include "./src/detection/YoloDetector.h"
#include "./src/detection/ArucoDetector.h"
#include "./src/wire/Wire.h"
#include "./src/servo/PCA9685.h"
#include "./src/servo/ServoKit.h"
#include "./src/util/util.h"
#include "./src/util/config.h"
#include "./src/network/SACAgent.h"
#include "./src/network/ReplayBuffer.h"
#include "./src/env/Env.h"
#include "./src/pid/PID.h"

// Threads
void panTiltThread(Utility::param* parameters);
void detectThread(Utility::param* parameters);
void syncThread(Utility::param* parameters);
void autoTuneThread(Utility::param* parameters);

pthread_cond_t trainCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t trainLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t sleepLock = PTHREAD_MUTEX_INITIALIZER;

pid_t pid = -1;

// Shared between threads
SACAgent* pidAutoTuner = nullptr;
TATS::Env* servos = nullptr;
cv::VideoCapture* camera = nullptr;
int additionalDelay = 0;

// Signal handlers
static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
volatile sig_atomic_t sig_value1;

int main(int argc, char** argv)
{

    std::srand( (unsigned)std::time( NULL ) );

    using namespace Utility;
    using namespace TATS;

    // Initialize defaults
    param* parameters = new Parameter();
    Config* config = new Config();
    servos = new TATS::Env();

    // TODO:: Make numParams a dynamic calculation
    // Create a shared memory buffer for experiance replay
    int numParams = 558298; // size of policy network! Will change if anything is edited in defaults
    boost::interprocess::shared_memory_object::remove("SharedMemorySegment");
    boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, "SharedMemorySegment", (sizeof(TD) * config->maxBufferSize + 1) + (sizeof(char) * numParams + 1));
    const ShmemAllocator alloc_inst(segment.get_segment_manager());
    SharedBuffer* sharedTrainingBuffer = segment.construct<SharedBuffer>("SharedBuffer")(alloc_inst);
    Utility::sharedString *s = segment.construct<Utility::sharedString>("SharedString")("", segment.get_segment_manager());

    // Setup stats dirs and remove old logs
    std::string losspath = "/trainingLoss.txt";
    std::string avePath = "/episodeAverages.txt";
    std::string stepPath = "/episodeStepRewards.txt";

    std::string path = get_current_dir_name();
    std::string statDir = "/stat/";
    std::string statPath = path + statDir;
    mkdir(statPath.c_str(), 0755);
    std::remove(( statPath + losspath ).c_str());

    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        std::string servoPath = statPath + std::to_string(servo);
        mkdir(servoPath.c_str(), 0755);
        
        std::remove(( servoPath + avePath ).c_str());
        std::remove(( servoPath + stepPath ).c_str());
    }

    // Parent process is image recognition PID/servo controller, second is SAC PID Autotuner
    if (config->multiProcess && config->trainMode) {
        pid = fork(); 
    } else {
        pid = getpid();
    }
    
    if (pid > 0) {

        parameters->pid = pid;

        // std::string pipeline = Utility::gstreamer_pipeline(0, 1920, 1080, 1920, 1080, 60, 2);
        // camera = new cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
        
        camera = new cv::VideoCapture(0, cv::CAP_GSTREAMER);
        // camera = new cv::VideoCapture(0, cv::CAP_V4L2);
        camera->set(cv::CAP_PROP_FRAME_WIDTH, config->captureSize[1]);
        camera->set(cv::CAP_PROP_FRAME_HEIGHT, config->captureSize[0]);
        // camera->set(cv::CAP_PROP_AUTOFOCUS, 0 );
        
        // Setup threads and PIDS
        pidAutoTuner = new SACAgent(config->numInput, config->numHidden, config->numActions, config->actionHigh, config->actionLow);
        if (!config->trainMode) {
            pidAutoTuner->eval();
        }
        
        // If autotuning is to be performed in this process
        if (config->multiProcess) {
            std::thread panTiltT(panTiltThread, parameters);
            std::thread detectT(detectThread, parameters);
            
            panTiltT.detach();
            detectT.detach();
            syncThread(parameters);

            // Terminate Child processes
            kill(pid, SIGQUIT);
            if (wait(NULL) != -1) {
                return 0;
            }
            else {
                return -1;
            }

        } else {
            std::thread panTiltT(panTiltThread, parameters);
            std::thread autoTuneT(autoTuneThread, parameters);
            std::thread detectT(detectThread, parameters);
            detectT.join();
            autoTuneT.join();
            panTiltT.join();
        }
        
    } else {
    
        // Kill child if parent killed
        prctl(PR_SET_PDEATHSIG, SIGKILL); 
 
        // SACAgent* pidAutoTunerChild = new SACAgent(config->numInput, config->numHidden, config->numActions, config->actionHigh, config->actionLow, true, 0.99, 5e-3, 0.2, 3e-4, 3e-4, 3e-4, device);
        SACAgent* pidAutoTunerChild = new SACAgent(config->numInput, config->numHidden, config->numActions, config->actionHigh, config->actionLow);

        // Variables for training
        int batchSize = config->batchSize;
        long maxTrainingSteps = config->maxTrainingSteps;
        long currentSteps = 0;
        int minBufferSize = config->minBufferSize;
        int maxBufferSize = config->maxBufferSize;
        int sessions = config->maxTrainingSessions;
        int numUpdates = config->numUpdates;
        double rate = config->trainRate;

        // Retrieve the training buffer from shared memory
        boost::interprocess::managed_shared_memory segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
        SharedBuffer* trainingBuffer = segment.find<SharedBuffer>("SharedBuffer").first;
        ReplayBuffer* replayBuffer = new ReplayBuffer(config->maxBufferSize, trainingBuffer);
        sharedString *s = segment.find_or_construct<sharedString>("SharedString")("", segment.get_segment_manager());

        // Setup signal mask
        struct sigaction sig_action;
        sigset_t oldmask;
        sigset_t newmask;
        sigset_t zeromask;

        memset(&sig_action, 0, sizeof(struct sigaction));

        sig_action.sa_flags = SA_SIGINFO;
        sig_action.sa_sigaction = usr_sig_handler1;

        sigaction(SIGHUP, &sig_action, NULL);
        sigaction(SIGINT, &sig_action, NULL);
        sigaction(SIGTERM, &sig_action, NULL);
        sigaction(SIGSEGV, &sig_action, NULL);
        sigaction(SIGUSR1, &sig_action, NULL);

        sigemptyset(&newmask);
        sigaddset(&newmask, SIGHUP);
        sigaddset(&newmask, SIGINT);
        sigaddset(&newmask, SIGTERM);
        sigaddset(&newmask, SIGSEGV);
        sigaddset(&newmask, SIGUSR1);

        sigprocmask(SIG_BLOCK, &newmask, &oldmask);
        sigemptyset(&zeromask);
        sig_value1 = 0;

        // Annealed ERE (Emphasizing Recent Experience)
        // https://arxiv.org/pdf/1906.04009.pdf
        double N0 = 0.996;
        double NT = 1.0;
        double T = maxTrainingSteps;
        double t_i = 0;

        // Wait on parent process' signal before training
        while ((sig_value1 != SIGINT) && (sig_value1 != SIGTERM) && (sig_value1 != SIGSEGV) && (sig_value1 != SIGTSTP))
        {
            sig_value1 = 0;

            // Sleep until signal is caught; train model on waking
        start:
            
            sigsuspend(&zeromask);

            if (sig_value1 == SIGUSR1) {

                std::cout << "Train signal received..." << std::endl;
                bool isTraining = true;

                // Begin training process
                while (isTraining) {
        
                    // Increment/set ERE related loop variables
                    double N = static_cast<double>(replayBuffer->size());
                    t_i += 1;
                    double n_i = N0 + (NT - N0) * (t_i / T);
                    int cmin = N - ( minBufferSize );
                        
                    // Check if training is over
                    if (currentSteps >= maxTrainingSteps) {
                        currentSteps = 0;
                        sessions--;
                        replayBuffer->clear();
                        t_i = 0;
                        goto start;
                    }
                    else {
                        currentSteps += 1;
                    }

                    // Perform a training session
                    for (int k = 0; k < numUpdates; k += 1) {
                        int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / numUpdates)), cmin);

                        TrainBuffer batch = replayBuffer->ere_sample(batchSize, startingRange);
                        pidAutoTunerChild->update(batch.size(), &batch);
                    }

                    // Write values to shared memory for parent to read
                    try {
                        pidAutoTunerChild->save_policy(s);
                    } catch (...) {
                        throw std::runtime_error("Cannot save policy params to shared memory array");
                    }
                    
                    // Inform parent new params are available
                    kill(getppid(), SIGUSR1);

                    // Sleep per train rate
                    long milis = static_cast<long>(1000.0 / rate);
                    msleep(milis);
                }
            }
        }
    }
}

void syncThread(Utility::param* parameters) {
    using namespace Utility;
    Utility::Config* config = new Utility::Config();

    // Retrieve model params array from shared memory 
    boost::interprocess::managed_shared_memory segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
    Utility::sharedString *s = segment.find_or_construct<Utility::sharedString>("SharedString")("", segment.get_segment_manager());

    // Setup signal mask
    sigset_t  mask;
    siginfo_t info;
    pid_t     child, p;
    int       signum;

    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGHUP);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGQUIT);
    sigaddset(&mask, SIGUSR1);
    sigaddset(&mask, SIGUSR2);

    if (sigprocmask(SIG_BLOCK, &mask, NULL) == -1) {
        throw std::runtime_error("Cannot block SIGUSR1 or SIGUSR2");
    }

    // Loop for syncing parent network params
    while (true) {
        
        // Wait until sync signal is received
        signum = sigwaitinfo(&mask, &info);
        if (signum == -1) {
            if (errno == EINTR)
                continue;
            std::runtime_error("Parent process: sigwaitinfo() failed");
        }

        // Update weights on signal received from autotune thread
        if (signum == SIGUSR1 && info.si_pid == pid) {
            try {
                pidAutoTuner->load_policy(s);
            } catch (...) {
                throw std::runtime_error("Cannot sync network parameters with auto-tune process");
            }
        }

        // Break when on SIGINT
        if (signum == SIGINT && !info.si_pid == parameters->pid) {
            std::cout << "Ctrl+C detected!" << std::endl;
            break;
        }
    }
}

static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context)
{
    // Take care of all segfaults
    if (sig_number == SIGSEGV || sig_number == SIGTSTP || sig_number == SIGINT)
    {
        camera->release();
        kill(getpid(), SIGKILL);
    }

    sig_value1 = sig_number;
}

// TODO: Seperate panTiltThread into a trainThread and evaluationThread functons....
void panTiltThread(Utility::param* parameters) {
    using namespace TATS;
    using namespace Utility;
    std::mt19937 eng{ std::random_device{}() };
    torch::Device device(torch::kCPU);
    std::default_random_engine _generator;
    std::normal_distribution<double> _distribution;

    auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
    Utility::Config* config = new Utility::Config();

    // Retrieve the training buffer from shared memory
    boost::interprocess::managed_shared_memory segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
    SharedBuffer* trainingBuffer = segment.find<SharedBuffer>("SharedBuffer").first;
    ReplayBuffer* replayBuffer = new ReplayBuffer(config->maxBufferSize, trainingBuffer, config->multiProcess);

    // Program loop variants
    unsigned int totalSteps = 0;
    bool recentlyReset = false;
    int doneCount = 0;

    // Train records
    int numEpisodes[NUM_SERVOS] = { 0 };
    double emaEpisodeRewardSum[NUM_SERVOS] = { 0.0 };
    double stepAverageRewards[NUM_SERVOS] = { 0.0 };
    double emaEpisodeStepSum[NUM_SERVOS] = { 0.0 };
    double totalEpisodeRewards[NUM_SERVOS] = { 0.0 };
    double totalEpisodeSteps[NUM_SERVOS] = { 0.0 };

    double totalEpisodeObjPredError[NUM_SERVOS] = { 0.0 };
    double stepAverageObjPredError[NUM_SERVOS] = { 0.0 };
    double emaEpisodeObjPredErrorSum[NUM_SERVOS] = { 0.0 };

    // training state variables
    bool isTraining = false;
    bool initialRandomActions = config->initialRandomActions;
    int numInitialRandomActions = config->numInitialRandomActions;
    bool stepsWithPretrainedModel = config->stepsWithPretrainedModel;
    int numTransferLearningSteps = config->numTransferLearningSteps;
    
    // Training data
    double predictedActions[NUM_SERVOS][NUM_ACTIONS] = { 0.0 };
    double stateArray[NUM_INPUT] = { 0.0 };
    SD currentState[NUM_SERVOS] = {{}};
    TD trainData[NUM_SERVOS] = {{}};
    ED eventData[NUM_SERVOS] = {{}};
    RD resetResults = {};
    SR stepResults = {};

    double rate = static_cast<double>(config->updateRate);

    try {
        resetResults = servos->reset();	
    } catch (...) {
        throw std::runtime_error("cannot reset servos");
    }
    
    for (int servo = 0; servo < NUM_SERVOS; servo++) {
        currentState[servo] = resetResults.servos[servo];
    }

    while (true) {
        
        if (config->useAutoTuning) {

            if (!servos->isDone()) {
                recentlyReset = false;
                doneCount = 0;
    
                for (int i = 0; i < NUM_SERVOS; i++) {

                    // Query network and get PID gains
                    if (config->trainMode) {
                        
                        if (initialRandomActions && numInitialRandomActions > 0) {
                            for (int a = 0; a < config->numActions; a++) {
                                predictedActions[i][a] = std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                            }
                        } else if (stepsWithPretrainedModel && numTransferLearningSteps > 0) {
                            currentState[i].getStateArray(stateArray);
                            at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, config->numInput }, options), true);
                            actions.to(torch::kCPU);
                            if (config->numActions > 1) {
                                auto actions_a = actions.accessor<double, 1>();
                                for (int a = 0; a < config->numActions; a++) {
                                    predictedActions[i][a] = actions_a[a];
                                }
                            }
                            else {
                                predictedActions[i][0] = actions.item().toDouble();
                            }
                        }
                        else {

                            if (initialRandomActions) {
                                initialRandomActions = false;
                                std::cout << "WARNING: Done generating initial random actions" << std::endl;
                            } else if (stepsWithPretrainedModel) {
                                stepsWithPretrainedModel = false;
                                std::cout << "WARNING: Done taking steps with pre-trained model in evaluation mode" << std::endl;
                            }
                            
                            // Perform Inference, get action(s) 
                            currentState[i].getStateArray(stateArray);
                            at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, config->numInput }, options), true);
                            actions.to(torch::kCPU);
                            if (config->numActions > 1) {
                                auto actions_a = actions.accessor<double, 1>();
                                for (int a = 0; a < config->numActions; a++) {
                                    predictedActions[i][a] = actions_a[a];
                                }
                            }
                            else {
                                predictedActions[i][0] = actions.item().toDouble();
                            }
                            
                        }
                    }
                    else {
                        currentState[i].getStateArray(stateArray);
                        at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, config->numInput }, options), false);
                        actions.to(torch::kCPU);
                        if (config->numActions > 1) {
                            auto actions_a = actions.accessor<double, 1>();
                            for (int a = 0; a < config->numActions; a++) {
                                predictedActions[i][a] = actions_a[a];
                            }
                        }
                        else {
                            predictedActions[i][0] = actions.item().toDouble();
                        }
                    }
                }

                try {
                    stepResults = servos->step(predictedActions, true, rate);		
                    totalSteps = (totalSteps + 1) % INT_MAX; 
                } catch(const std::exception& e) {
                    std::cerr << e.what() << std::endl;
                    throw std::runtime_error("cannot step with servos");
                }
                
                bool reset = false;
                bool updated = false;
                for (int servo = 0; servo < NUM_SERVOS; servo++) {

                    // Next state becomes the current state next time we step
                    trainData[servo] = stepResults.servos[servo];
                    trainData[servo].currentState = currentState[servo];
                    currentState[servo] = trainData[servo].nextState;

                    // If servo is disabled, null record
                    if (trainData[servo].empty) {
                        continue;
                    }
                    
                    if (config->trainMode) {

                        if (initialRandomActions && !updated) {
                            updated = true;
                            numInitialRandomActions--;
                        } else if (stepsWithPretrainedModel && !updated) {
                            updated = true;
                            numTransferLearningSteps--;
                        }

                        // If early episode termination
                        if (config->episodeEndCap) {
                            if (totalEpisodeSteps[servo] > config->maxStepsPerEpisode) {
                                reset = true;
                            }
                        }

                        // Add to replay buffer for training
                        if (replayBuffer->size() <= config->maxBufferSize) {
                            replayBuffer->add(trainData[servo]);
                        }

                        // EMA of steps and rewards (With 30% weight to new episodes; or 5 episode averaging)
                        std::string path = get_current_dir_name();
                        double percentage = (1.0 / 3.0);
                        double timePeriods = (2.0 / percentage) - 1.0;
                        double emaWeight = (2.0 / (timePeriods + 1.0));

                        // Data logging
                        if (trainData[servo].done || reset) {
                            
                            numEpisodes[servo] += 1;
                            emaEpisodeRewardSum[servo] = (totalEpisodeRewards[servo] - emaEpisodeRewardSum[servo]) * emaWeight + emaEpisodeRewardSum[servo];
                            emaEpisodeStepSum[servo] = (totalEpisodeSteps[servo] - emaEpisodeStepSum[servo]) * emaWeight + emaEpisodeStepSum[servo];
                            emaEpisodeObjPredErrorSum[servo] = (totalEpisodeObjPredError[servo] - emaEpisodeObjPredErrorSum[servo]) * emaWeight + emaEpisodeObjPredErrorSum[servo];
                
                            // Log Episode averages
                            std::string episodeData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                                                    (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                                                    + std::to_string(totalSteps) + ','
                                                    + std::to_string(numEpisodes[servo]) + ','
                                                    + std::to_string(totalEpisodeSteps[servo] > 0.0 ? totalEpisodeRewards[servo] / totalEpisodeSteps[servo] : 0.0) + ','
                                                    + std::to_string(totalEpisodeSteps[servo]) + ','
                                                    + std::to_string(emaEpisodeRewardSum[servo] / emaEpisodeStepSum[servo]) + ','
                                                    + std::to_string(emaEpisodeStepSum[servo]) + ','
                                                    + std::to_string(emaEpisodeObjPredErrorSum[servo] / emaEpisodeStepSum[servo]);
                            
                            appendLineToFile(path + "/stat/" + std::to_string(servo) + "/episodeAverages.txt", episodeData);

                            totalEpisodeSteps[servo] = 0.0;
                            totalEpisodeRewards[servo] = 0.0;
                            totalEpisodeObjPredError[servo] = 0.0;
                        } else {
                            // Average reward in a step
                            totalEpisodeSteps[servo] += 1.0;						
                            totalEpisodeRewards[servo] += trainData[servo].errors[0];
                            totalEpisodeObjPredError[servo] += trainData[servo].errors[1];

                            stepAverageRewards[servo] = (trainData[servo].errors[0] - stepAverageRewards[servo]) * emaWeight + stepAverageRewards[servo];
                            stepAverageObjPredError[servo] = (trainData[servo].errors[1] - stepAverageObjPredError[servo]) * emaWeight + stepAverageObjPredError[servo];

                            std::string stepData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                                                (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                                                + std::to_string(totalSteps) + ','
                                                + std::to_string(trainData[servo].errors[0]) + ',' 
                                                + std::to_string(stepAverageRewards[servo])  + ','
                                                + std::to_string(trainData[servo].errors[1])  + ','
                                                + std::to_string(stepAverageObjPredError[servo]);

                            appendLineToFile(path + "/stat/" + std::to_string(servo) + "/episodeStepRewards.txt", stepData);
                        
                        }   
                    }

                    // Debug output
                    if (config->logOutput) {
                        double state[NUM_INPUT];
                        std::cout << (servo ? "Pan step info:" : "Tilt step info:") << std::endl;
                        std::cout << "Next State: ";
                        trainData[servo].nextState.getStateArray(state);
                        for (int j = 0; j < NUM_INPUT; j++) {
                            std::cout << std::to_string(state[j]) << ", ";
                        }
                        std::cout << std::endl;

                        std::cout << "Current State: ";
                        trainData[servo].currentState.getStateArray(state);
                        for (int j = 0; j < NUM_INPUT; j++) {
                            std::cout << std::to_string(state[j]) << ", ";
                        }
                        std::cout << std::endl;

                        std::cout << "FPS Delay: ";
                        std::cout << std::to_string(additionalDelay);
                        std::cout << std::endl;

                        std::cout << "Reward: ";
                        std::cout << std::to_string(trainData[servo].reward);
                        std::cout << std::endl;

                        std::cout << "Done: ";
                        std::cout << std::to_string(trainData[servo].done);
                        std::cout << std::endl;

                        std::cout << "Actions: ";

                        if (config->usePIDs) {
                            for (int j = 0; j < NUM_ACTIONS; j++) {
                                std::cout << std::to_string(Utility::rescaleAction(trainData[servo].actions[j], config->actionLow, config->actionHigh)) << ", ";
                            }
                            
                        } else {
                            std::cout << std::to_string(Utility::rescaleAction(trainData[servo].actions[0], config->actionLow, config->actionHigh)) << ", ";

                            if (config->usePOT) {
                                std::cout << std::to_string(Utility::rescaleAction(trainData[servo].actions[1], 0.0, config->dims[servo])) << std::endl;
                            }

                            std::cout << "Errors: ";
                            std::cout << std::to_string(trainData[servo].errors[0]) << " "; 
                            if (config->usePOT) {
                                std::cout << std::to_string(trainData[servo].errors[1]);
                            }
                            std::cout << std::endl;
                        }
                         std::cout << std::endl;
                       
                    }    

                    // Start training thread/processes when ready
                    if (config->trainMode) {
                        // Inform child process to start training
                        if (config->multiProcess) {
                            if (!isTraining && replayBuffer->size() > (config->minBufferSize)) {
                                std::cout << "Sending train signal..." << std::endl;
                                isTraining = true;
                                kill(parameters->pid, SIGUSR1);
                            } 
                        } else if (!isTraining && replayBuffer->size() > (config->minBufferSize)) {
                            // Inform autotune thread start training 
                            pthread_mutex_lock(&trainLock);
                            pthread_cond_broadcast(&trainCond);
                            pthread_mutex_unlock(&trainLock);
                        }
                    }
                }

                if (reset) { goto reset; }   
            }
            else if (!recentlyReset) {
                doneCount++;
                recentlyReset = true;

                reset:

                // Vary env sync rate and FPS to simulate different latency configurations
                if (config->trainMode) {
                    rate = static_cast<double>(config->updateRate);
                    double adjustment =  (( rate ) / 2.0) * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                    rate += adjustment;

                    // Locked to avoid race condition on additionalDelay with detectThread
                    pthread_mutex_lock(&sleepLock);
                    if (config->variableFPS) {
                        additionalDelay = static_cast<int>(std::round(config->FPSVariance * std::uniform_real_distribution<double>{ 0, 1.0 }(eng)));
                    } 
                    pthread_mutex_unlock(&sleepLock);
                }
                
                // Vary angle of reset after set amount of time to enable AI to work at any reset angle
                if (config->trainMode && config->varyResetAngles) {
            
                    double newAngles[NUM_SERVOS] = { 0.0 };
                    for (int servo = 0; servo < NUM_SERVOS; servo++) {
                        newAngles[servo] = config->resetAngleVariance * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                    }
                    resetResults = servos->reset(newAngles);
 
                } else { 
                    resetResults = servos->reset(config->useCurrentAngleForReset);
                }
                
                // Hold onto reset results
                for (int servo = 0; servo < NUM_SERVOS; servo++) {
                    currentState[servo] = resetResults.servos[servo];
                }   		
            } else {
                doneCount++;

                if (doneCount >= config->resetAfterNInnactiveFrames && config->resetAfterNInnactiveFrames > 0) {
                    if (config->trainMode) {
                        double newAngles[NUM_SERVOS] = { 0.0 };
                        for (int servo = 0; servo < NUM_SERVOS; servo++) {
                            newAngles[servo] = config->resetAngleVariance * std::uniform_real_distribution<double>{ -1.0, 1.0 }(eng);
                        }
                        resetResults = servos->reset(newAngles);
                    } else {
                        resetResults = servos->reset();            
                    }
                    
                    doneCount = 0;
                    
                } else {
                    resetResults = servos->reset(config->useCurrentAngleForReset);            
                }
                
                // Hold onto reset results
                for (int servo = 0; servo < NUM_SERVOS; servo++) {
                    currentState[servo] = resetResults.servos[servo];
                }
            }    
        }
        else {
            if (!servos->isDone()) {
                for (int i = 0; i < NUM_SERVOS ; i++) {
                    predictedActions[i][0] = config->defaultGains[0];
                    predictedActions[i][1] = config->defaultGains[1];
                    predictedActions[i][2] = config->defaultGains[2];
                }

                try {
                    stepResults = servos->step(predictedActions, false);

                    double state[NUM_INPUT];
                    for (int i = 0; i < NUM_SERVOS ; i++) {
                        
                        if (stepResults.servos[i].empty) {
                            continue;
                        }
                    }
                    
                } catch (...) {
                    throw std::runtime_error("cannot step with servos");
                }
            }
            else {

                try {
                    resetResults = servos->reset(config->useCurrentAngleForReset);
                } catch (...) {
                    throw std::runtime_error("cannot reset servos");
                }
            }	
        }
    }
}

void detectThread(Utility::param* parameters)
{
    using namespace TATS;
    using namespace Utility;

    Utility::Config* config = new Utility::Config();

    // Setup Camera
    if (!camera->isOpened())
    {
        throw std::runtime_error("cannot initialize camera");
    } else {
        if (GetImageFromCamera(camera).empty())
        {
            throw std::runtime_error("Issue reading frame!");
        }
    }

    // user hyperparams
    int frameCount = 0;
    int recheckFrequency = config->recheckFrequency;
    int trackerType = config->trackerType;
    bool useTracking = config->useTracking;
    bool draw = config->draw;
    bool showVideo = config->showVideo;

    // program state variables
    bool programStart = true;
    bool rechecked = false;
    bool isTracking = false;
    bool isSearching = false;
    int lossCount = 0;
    int trackCount = 0;
    int searchCount = 0;
    int searchCountMax = 5;
    int lossCountMax = config->lossCountMax;
    std::vector<std::string> targets = config->targets;
    std::string currentTarget = "";

    // Create object tracker to optimize detection performance
    cv::Rect2i roi;
    cv::Rect2d boundingBox;
    cv::Ptr<cv::Tracker> tracker;
    if (useTracking) {
        tracker = createOpenCVTracker(trackerType);
    }

    // Object center coordinates
    int frameCenterX = 0;
    int frameCenterY = 0;

    // Object coordinates
    int objX = 0;
    int objY = 0;

    cv::Mat frame;
    cv::Mat detection;
    auto execbegin = std::chrono::high_resolution_clock::now();

    if (!camera->isOpened()) {
        throw std::runtime_error("cannot initialize camera");
    }

    // load the network
    Detect::ObjectDetector* detector;
    std::string path = get_current_dir_name();
    std::string weights = path + config->detectorPath;
    std::vector<std::string> class_names = config->classes;

    switch(config->detector) {
        case Utility::DetectorType::CASCADE:
            detector = new Detect::CascadeDetector(weights);
            break; 
        case Utility::DetectorType::ARUCO:
            detector = new Detect::ArucoDetector();
            break; 
        case Utility::DetectorType::YOLO:
            detector = new Detect::YoloDetector(weights, class_names);
            break; 
        default : 
            detector = new Detect::CascadeDetector(weights);
    }

    std::vector<struct Detect::DetectionData> results;
    while (true) {

        std::chrono::steady_clock::time_point detectLoopStart = std::chrono::steady_clock::now(); // For delay
        
        // Vary FPS during training. 
        if (config->trainMode && config->variableFPS) {
            pthread_mutex_lock(&sleepLock);
            int sleepTime = 0;
            sleepTime = additionalDelay;
            pthread_mutex_unlock(&sleepLock);
            
            if (sleepTime > 0) {
                // Note: Anything less than 10 milliseconds doesn't register well with std::chrono::steady_clock
                Utility::msleep(sleepTime);
            }
        }

        if (isSearching) {
            searchCount += 1;
            
            // TODO:: Perform better search ruetine
            // For now servo thread detects when done and sends a reset command to servos
        }

        try
        {
            try {
                // crop the image to generate equal setpoints for PIDs
                frame = GetImageFromCamera(camera);
                
                int cropSize = config->dims[0];
                int offsetW = (frame.cols - cropSize) / 2;
                int offsetH = (frame.rows - cropSize) / 2;
                cv::Rect region(offsetW, offsetH, cropSize, cropSize);
                frame = frame(region).clone();
            } catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                camera->release();
                throw std::runtime_error("could not get image from camera");
            }

            if (!useTracking) {
                goto detect;
            }

            if (isTracking) {
                
                isSearching = false;

                // Get the new tracking result
                if (!tracker->update(frame, boundingBox)) {
            lostTracking:
                    isTracking = false;
                    frameCount = 0.0;
                    lossCount++;
                    trackCount = 0;
                    currentTarget = "";
                    goto detect;
                }

                roi = boundingBox;

                // Chance to revalidate object tracking quality
                if (frameCount >= recheckFrequency) {
                    frameCount = 0;
                    rechecked = true;
                    goto detect;
                } else {
                    frameCount++;
                }

            validated:
                // Determine object and frame centers
                frameCenterX = frame.cols / 2;
                frameCenterY = frame.rows / 2;
                objX = roi.x + (roi.width / 2);
                objY = roi.y + (roi.height / 2);

                // Enter State data
                auto wall_now = std::chrono::high_resolution_clock::now();
                double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - execbegin).count() * 1e-9;
                
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                double spf = std::chrono::duration<double>(now - detectLoopStart).count();
                
                ED eventDataArray[2] {
                    {
                        false,
                        static_cast<double>(frameCenterY - objY),
                        static_cast<double>(frameCenterY),
                        static_cast<double>(roi.height),
                        static_cast<double>(roi.y),
                        static_cast<double>(objY),
                        timestamp,
                        spf
                    },
                    {
                        false,
                        static_cast<double>(frameCenterX - objX),
                        static_cast<double>(frameCenterX),
                        static_cast<double>(roi.width),
                        static_cast<double>(roi.x),
                        static_cast<double>(objX),
                        timestamp,
                        spf
                    }
                };

                try {
                    servos->update(eventDataArray);
                } catch (...) {
                    throw std::runtime_error("cannot update event data");
                }

                // draw to frame
                if (draw) {
                    Utility::drawPred(1.0, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height, frame);
                }
            }
            else {

            detect:

                try {
                    std::vector<struct Detect::DetectionData> out = detector->detect(frame, draw, 1);
                    results = out;
                } catch (const std::exception& e)
                {
                    std::cerr << e.what() << std::endl;
                    camera->release();
                    throw std::runtime_error("Could not detect target from frame");
                }
                
                Detect::DetectionData result;
                result.found = false;
  
                if (!results.empty()) { 
                    if (programStart || isSearching) {
                        for (auto res : results) {
                            
                            // Pick the first one we see
                            if (std::find(targets.begin(), targets.end(), res.target) != targets.end()) {
                                result = res;
                                roi = result.boundingBox;
                                currentTarget = result.target;
                                break;
                            }
                        }         
                    } else if (useTracking && rechecked) {
                        rechecked = false;
                        for (auto res : results) {
                            result = res;
                            
                            // TODO::Determine criteria by which OpenCV Tracking is still accurate
                            // If they intersect
                            // if (((result.boundingBox & roi).area() > 0.0)) {
                            goto validated; 
                            // }  
                        }
                        
                        goto lostTracking;
                    } else {                
                        if (results.size() > 1) {
                            
                            int bestDistance = config->dims[0];
                            for (auto res : results) {
                                // TODO:: Assumption below potentially wrong, use POT in the future
                                if (currentTarget == res.target) {                                

                                    cv::Point2i a = res.center;
                                    cv::Point2i b = (roi.tl() + roi.br()) / 2;
                                    int distance = std::sqrt<int>( (a.x-b.x) * (a.x-b.x) + (a.y-b.y) * (a.y-b.y) );

                                    if (distance <= bestDistance) {
                                        bestDistance = distance;
                                        result = res;
                                        roi = result.boundingBox;
                                    }
                                }
                            } 
                        } else {
                            if (currentTarget == results[0].target) {
                                result = results[0];
                                roi = result.boundingBox;
                            } 
                        }
                    }
                } 
                
                // If we confidently found the object we are looking for
                if (result.found) {

                    // Update loop variants
                    programStart = false;
                    lossCount = 0;
                    trackCount = (trackCount + 1) % CHAR_MAX; 
                    isSearching = false;

                    // Determine object and frame centers
                    frameCenterX = frame.cols / 2;
                    frameCenterY = frame.rows / 2;
                    objX = result.center.x;
                    objY = result.center.y;
                    
                    // Fill out state data
                    auto wall_now = std::chrono::high_resolution_clock::now();
                    double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - execbegin).count() * 1e-9;
                    
                    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                    std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                    double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                    ED eventDataArray[2] {
                        {
                            false,
                            static_cast<double>(frameCenterY - objY),
                            static_cast<double>(frameCenterY),
                            static_cast<double>(roi.height),
                            static_cast<double>(roi.y),
                            static_cast<double>(objY),
                            timestamp,
                            spf
                        },
                        {
                            false,
                            static_cast<double>(frameCenterX - objX),
                            static_cast<double>(frameCenterX),
                            static_cast<double>(roi.width),
                            static_cast<double>(roi.x),
                            static_cast<double>(objX),
                            timestamp,
                            spf
                        }
                    };

                    try {
                        servos->update(eventDataArray);
                    } catch (...) {
                        throw std::runtime_error("cannot update event data");
                    }

                    if (useTracking && !isTracking) {

                        try {
                            tracker = createOpenCVTracker(trackerType);
                            if (tracker->init(frame, roi)) {
                                isTracking = true;
                            } 
                        } catch (const std::exception& e) {
                            std::cerr << e.what() << std::endl;
                            camera->release();
                            throw std::runtime_error("Could not init opencv tracker");
                        }
                    }
                }
                else if (!config->trainMode && config->usePOT) {
                    lossCount++;
                    frameCount = 0.0;
                    rechecked = false;

                    // Target is out of sight, inform PID's, model, and servos
                    if (lossCount >= lossCountMax) {
                        trackCount = 0;
                        isSearching = true;
                        isTracking = false;
                        lossCount = 0;

                        // Enter state data
                        frameCenterX = frame.cols / 2;
                        frameCenterY = frame.rows / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        ED eventDataArray[2] {
                            {
                                true,
                                static_cast<double>(frameCenterY),
                                static_cast<double>(frameCenterY),
                                0.0,
                                0.0,
                                static_cast<double>(frameCenterY * 2), // Max error
                                timestamp,
                                spf
                            },
                            {
                                true,
                                static_cast<double>(frameCenterX),
                                static_cast<double>(frameCenterX),
                                0.0,
                                0.0,
                                static_cast<double>(frameCenterX * 2),
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    } else if (!isSearching && !programStart && trackCount > 3) {
                        // Get the prediction location of object relative to frame center
                        if (config->logOutput) {
                            std::cout << "Using Predictive Object Tracking: frame # " << std::to_string(lossCount) << std::endl;
                        }
                        
                        // Enter state data
                        double locations[NUM_SERVOS] = { 0.0 };
                        servos->getPredictedObjectLocation(locations);
                        frameCenterX = config->dims[1] / 2;
                        frameCenterY = config->dims[0] / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        ED eventDataArray[2] {
                            {
                                false,
                                static_cast<double>(frameCenterY) - locations[0],
                                static_cast<double>(frameCenterY),
                                0.0,
                                0.0,
                                locations[0], // Predicted location
                                timestamp,
                                spf
                            },
                            {
                                false,
                                static_cast<double>(frameCenterX) - locations[1],
                                static_cast<double>(frameCenterX),
                                0.0,
                                0.0,
                                locations[1],
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    }
                }
                else {
                    lossCount++;
                    rechecked = false;

                    // Target is out of sight, inform PID's, model, and servos
                    if (lossCount >= lossCountMax) {
                        
                        isSearching = true;
                        isTracking = false;
                        lossCount = 0;
                        trackCount = 0;

                        // Enter State data
                        frameCenterX = frame.cols / 2;
                        frameCenterY = frame.rows / 2;
                        
                        auto wall_now = std::chrono::high_resolution_clock::now();
                        double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - execbegin).count() * 1e-9;
                        
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::duration elapsed = now - detectLoopStart;
                        double spf = std::chrono::duration<double>(now - detectLoopStart).count();

                        ED eventDataArray[2] {
                            {
                                true,
                                static_cast<double>(frame.rows), // Obj not on screen, Max error
                                static_cast<double>(frameCenterY),
                                0.0,
                                0.0,
                                static_cast<double>(frameCenterY * 2), 
                                timestamp,
                                spf
                            },
                            {
                                true,
                                static_cast<double>(frame.cols),
                                static_cast<double>(frameCenterX),
                                0.0,
                                0.0,
                                static_cast<double>(frameCenterX * 2),
                                timestamp,
                                spf
                            }
                        };
                        
                        try {
                            servos->update(eventDataArray);
                        } catch (...) {
                            throw std::runtime_error("cannot update event data");
                        }
                    } 
                }
            }

            if (showVideo) {
                cv::imshow("Viewport", frame);
                cv::waitKey(1);
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            camera->release();
            throw std::runtime_error("Issue detecting target from video");
        }
    }
}

void autoTuneThread(Utility::param* parameters)
{
    using namespace TATS;
    using namespace Utility;
    Utility::Config* config = new Utility::Config();
    
    // Setup replay buff logging
    std::string path = get_current_dir_name();
    std::string statDir = "/stat/";
    std::string statPath = path + statDir;
    std::string logPath = statPath + "/replayBuffStats.txt";

    if (fileExists(logPath)) {
        std::remove(logPath.c_str());
    }

    // Variables for training
    int batchSize = config->batchSize;
    long maxTrainingSteps = config->maxTrainingSteps;
    long currentSteps = 0;
    int minBufferSize = config->minBufferSize;
    int maxBufferSize = config->maxBufferSize;
    int sessions = config->maxTrainingSessions;
    int numUpdates = config->numUpdates;
    double rate = config->trainRate;

    // Annealed ERE (Emphasizing Recent Experience)
    // https://arxiv.org/pdf/1906.04009.pdf
    double N0 = 0.996;
    double NT = 1.0;
    double T = maxTrainingSteps;
    double t_i = 0;
    
    // Retrieve the training buffer from shared memory
    boost::interprocess::managed_shared_memory segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
    SharedBuffer* trainingBuffer = segment.find<SharedBuffer>("SharedBuffer").first;
    ReplayBuffer* replayBuffer = new ReplayBuffer(config->maxBufferSize, trainingBuffer, config->multiProcess);

start:

    replayBuffer->clear();
    
    if (sessions <= 0) {
        std::cout << "Training session over!!" << std::endl;
        return;
    }
    
    // Wait for the buffer to fill
    pthread_mutex_lock(&trainLock);
    while (replayBuffer->size() <= config->minBufferSize) {
        pthread_cond_wait(&trainCond, &trainLock);
    }
    pthread_mutex_unlock(&trainLock);

    while (true) {

        double N = static_cast<double>(replayBuffer->size());
        t_i += 1;
        double n_i = N0 + (NT - N0) * (t_i / T);
        int cmin = N - ( minBufferSize );
        
        // Check if training is over
        if (currentSteps >= maxTrainingSteps) {
            currentSteps = 0;
            sessions--;
            replayBuffer->clear();
            t_i = 0;
            goto start;
        }
        else {
            currentSteps += 1;
        }

        for (int k = 0; k < numUpdates; k += 1) {
            int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / numUpdates)), cmin);

            TrainBuffer batch = replayBuffer->ere_sample(batchSize, startingRange);
            pidAutoTuner->update(batch.size(), &batch);
            
            std::string stepData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                                                 (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                                                 + std::to_string(currentSteps) + ','
                                                 + std::to_string(startingRange) + ',' 
                                                 + std::to_string(N);

            appendLineToFile(logPath, stepData);
        }

        long milis = static_cast<long>(1000.0 / rate);
        Utility::msleep(milis);
    }
}