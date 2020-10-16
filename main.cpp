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

// Pytorch imports
#include <torch/torch.h>

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
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
#include "./src/wire/Wire.h"
#include "./src/servo/PCA9685.h"
#include "./src/servo/ServoKit.h"
#include "./src/util/util.h"
#include "./src/util/data.h"
#include "./src/network/SACAgent.h"
#include "./src/network/ReplayBuffer.h"
#include "./src/env/Env.h"
#include "./src/pid/PID.h"

// Threads
void* panTiltThread(void* args);
void* detectThread(void* args);
void* autoTuneThread(void* args);

// Thread Sync
pthread_mutex_t stateDataLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t trainCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t trainLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t dataCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t dataLock = PTHREAD_MUTEX_INITIALIZER;

SACAgent* pidAutoTuner = nullptr;
TATS::Env* servos = nullptr;

// Camera related initializations
std::string pipeline = Utility::gstreamer_pipeline(1280, 720, 1280, 720, 60, 0);
cv::VideoCapture camera(pipeline, cv::CAP_GSTREAMER);

// Log files
std::string statFileName = "/stat/episodeAverages.txt";
std::string lossFileName = "/stat/trainingLoss.txt";
std::string stateDir = "/stat";

// Signal handlers
static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
volatile sig_atomic_t sig_value1;


int main(int argc, char** argv)
{
	using namespace Utility;
	using namespace TATS;
	
    // Initialize defaults and params
    param* parameters = (param*)malloc(sizeof(param));
	parameters->config = new Config();
    servos = new TATS::Env(parameters, &dataLock, &dataCond);

    // Create a large array for syncing parent and child process' model parameters
	int ShmID = shmget(IPC_PRIVATE, 10000 * sizeof(double), IPC_CREAT | 0666);
	if (ShmID < 0) {
        throw std::runtime_error("Could not initialize shared memory");
	}

	// Create a shared memory buffer for experiance replay
	boost::interprocess::shared_memory_object::remove("SharedMemorySegment");
	boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, "SharedMemorySegment", sizeof(TD) * parameters->config->maxBufferSize + 1);
	const ShmemAllocator alloc_inst(segment.get_segment_manager());
	SharedBuffer* sharedTrainingBuffer = segment.construct<SharedBuffer>("SharedBuffer") (alloc_inst);

    // Setup stats
    std::string avePath = get_current_dir_name() + statFileName;
	std::string lossPath = get_current_dir_name() + lossFileName;
	std::string statPath = get_current_dir_name() + stateDir;

	mkdir(statPath.c_str(), 0755);

	if (fileExists(avePath)) {
		std::remove(avePath.c_str());
	}

	if (fileExists(lossPath)) {
		std::remove(lossPath.c_str());
	}
    
    // Check for CUDA
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
    }

    // Setup Camera
    if (!camera.isOpened())
	{
		throw std::runtime_error("cannot initialize camera");
	} else {
        cv::Mat test = GetImageFromCamera(camera);

        if (test.empty())
        {
            throw std::runtime_error("Issue reading frame!");
        }

        int height = test.rows;
        int width = test.cols;

        parameters->dims[1] = width;
        parameters->dims[0] = height;
    }

    /*
        // Uno class names
        std::vector<std::string> class_names = {
            "0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"
        };

        // load the network
        std::string path = get_current_dir_name();
        std::string weights = path + "/models/yolo/yolo_uno.torchscript.pt";
        Detect::YoloDetector detector = Detect::YoloDetector(weights, class_names);

        // Detect loop
        while (true) {
            cv::Mat input_image = GetImageFromCamera(camera);
            if (input_image.empty()) {
                std::cout << "Issue getting frame from camera!!" << std::endl;
                continue;
            }

            auto results = detector.detect(input_image, true, 1);
            cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
            cv::imshow("Result", input_image);
            cv::waitKey(1);
        }

    */

    // Setup pids
	parameters->rate = parameters->config->updateRate; 
	parameters->isTraining = false;
	parameters->freshData = false;

    // Parent process is image recognition PID/servo controller, second is SAC Servo autotuner
    pid_t pid = fork();
	if (pid > 0) {
        
        // Find shared memory reference for the parent
		double* ShmPTRParent;
		ShmPTRParent = (double*)shmat(ShmID, 0, 0);

		if (ShmPTRParent == nullptr) {
			throw std::runtime_error("Could not initialize shared memory");
		}

        parameters->pid = pid;
        
        // Kill child if parent dies
		prctl(PR_SET_PDEATHSIG, SIGKILL); 

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


        // Setup threads and PIDS
        PID* pan = new PID(parameters->config->defaultGains[0], parameters->config->defaultGains[1], parameters->config->defaultGains[2], parameters->config->pidOutputLow, parameters->config->pidOutputHigh, static_cast<double>(parameters->dims[1]) / 2.0);
        PID* tilt = new PID(parameters->config->defaultGains[0], parameters->config->defaultGains[1], parameters->config->defaultGains[2], parameters->config->pidOutputLow, parameters->config->pidOutputHigh, static_cast<double>(parameters->dims[0]) / 2.0);
        parameters->pan = pan;
        parameters->tilt = tilt;
        pidAutoTuner = new SACAgent(parameters->config->numInput, parameters->config->numHidden, parameters->config->numActions, parameters->config->actionHigh, parameters->config->actionLow);
        
        pthread_t detectTid, panTiltTid;
		pthread_create(&panTiltTid, NULL, panTiltThread, (void*)parameters);
		pthread_create(&detectTid, NULL, detectThread, (void*)parameters);
		pthread_detach(panTiltTid);
		pthread_detach(detectTid);

        // Loop for syncing parent network params
        while (true) {

			signum = sigwaitinfo(&mask, &info);
			if (signum == -1) {
				if (errno == EINTR)
					continue;
				std::runtime_error("Parent process: sigwaitinfo() failed");
			}

			// Update weights on signal received from autotune thread
			if (signum == SIGUSR1 && info.si_pid == pid) {
				std::cout << "Loading new weights..." << std::endl;
				if (pthread_mutex_lock(&stateDataLock) == 0) {
					int valuesRead = pidAutoTuner->sync(true, ShmPTRParent);
					std::cout << "Tensors read in parent: " << valuesRead << std::endl;
				}

				pthread_mutex_unlock(&stateDataLock);
			}

			// Break when on SIGINT
			if (signum == SIGINT && !info.si_pid == pid) {
				camera.release();
				std::cout << "Ctrl+C detected!" << std::endl;
				break;
			}
		}

		// Terminate Child processes
		kill(-pid, SIGQUIT);
		if (wait(NULL) != -1) {
			return 0;
		}
		else {
			return -1;
		}

    } else {
        SACAgent* pidAutoTunerchild = new SACAgent(parameters->config->numInput, parameters->config->numHidden, parameters->config->numActions, parameters->config->actionHigh, parameters->config->actionLow);

        // Get Shared memory reference for the child
		double* ShmPTRChild;
		ShmPTRChild = (double*)shmat(ShmID, 0, 0);

		if (ShmPTRChild == nullptr) {
			throw std::runtime_error("Could not initialize shared memory");
		}

        // Retrieve the training buffer from shared memory
		boost::interprocess::managed_shared_memory _segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
		SharedBuffer* trainingBuffer = _segment.find<SharedBuffer>("SharedBuffer").first;
		ReplayBuffer* replayBuffer = new ReplayBuffer(parameters->config->maxBufferSize, trainingBuffer);


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

		// Wait on parent process' signal before training
		while ((sig_value1 != SIGINT) && (sig_value1 != SIGTERM))
		{
			sig_value1 = 0;

			// Sleep until signal is caught; train model on waking
			sigsuspend(&zeromask);

			if (sig_value1 == SIGUSR1) {

				std::cout << "Train signal received..." << std::endl;

				bool isTraining = true;

				// Begin training process
				while (isTraining) {

			    	// replayBuffer->removeOld(batchSize);
                    // TrainBuffer batch = replayBuffer->sample(batchSize);
                    // pidAutoTunerChild->update(batchSize, &batch);
					// int valuesWritten = pidAutoTunerChild->sync(false, ShmPTRChild);
					// std::cout << "Tensors written in child: " << valuesWritten << std::endl;

					kill(getppid(), SIGUSR1);

					
					// if (replayBuffer->size() == 0) {
					// 	isTraining = false; 
					// }
					// else {
					// 	msleep(milis);
					// }			
				}
			}
		}
    }
}

static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context)
{
	// Take care of all segfaults
	if (sig_number == SIGSEGV)
	{
		perror("SIGSEV: Address access error.");
		exit(-1);
	}

	sig_value1 = sig_number;
}

void* panTiltThread(void* args) {
	using namespace TATS;
	using namespace Utility;
	std::mt19937 eng{ std::random_device{}() };
    torch::Device device(torch::kCUDA);
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	param* parameters = (param*)args;

    // Retrieve the training buffer from shared memory
	boost::interprocess::managed_shared_memory _segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
	SharedBuffer* trainingBuffer = _segment.find<SharedBuffer>("SharedBuffer").first;
	ReplayBuffer* replayBuffer = new ReplayBuffer(parameters->config->maxBufferSize, trainingBuffer);

	// Training options and record keeping
	double episodeAverageRewards = 0.0;
	double episodeAverageSteps = 0.0;
	int numEpisodes = 0;
	double episodeRewards = 0.0;
	double episodeSteps = 0.0;

	// training state variables
	bool initialRandomActions = parameters->config->initialRandomActions;
	int numInitialRandomActions = parameters->config->numInitialRandomActions;
	
	double predictedActions[NUM_SERVOS][NUM_ACTIONS];
	double stateArray[NUM_INPUT];
	SD currentState[NUM_SERVOS];
	TD trainData[NUM_SERVOS];
	ED eventData[NUM_SERVOS];
	RD resetResults;

	resetResults = servos->reset();	
	
	for (int servo = 0; servo < NUM_SERVOS; servo++) {
		currentState[servo] = resetResults.servos[servo];
	}

	while (true) {

		if (parameters->config->useAutoTuning) {

			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {

					// Query network and get PID gains
					if (parameters->config->trainMode) {
						if (initialRandomActions && numInitialRandomActions >= 0) {

							numInitialRandomActions--;
							std::cout << "Random action count: " << numInitialRandomActions << std::endl;
							for (int a = 0; a < parameters->config->numActions; a++) {
								predictedActions[i][a] = std::uniform_real_distribution<double>{ parameters->config->actionLow, parameters->config->actionHigh }(eng);;
							}
						}
						else {
							currentState[i].getStateArray(stateArray);
							at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options), true);

							if (parameters->config->numActions > 1) {
								auto actions_a = actions.accessor<double, 1>();
								for (int a = 0; a < parameters->config->numActions; a++) {
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
						at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, parameters->config->numInput }, options), false);
						if (parameters->config->numActions > 1) {
							auto actions_a = actions.accessor<double, 1>();
							for (int a = 0; a < parameters->config->numActions; a++) {
								predictedActions[i][a] = actions_a[a];
							}
						}
						else {
							predictedActions[i][0] = actions.item().toDouble();
						}
					}
					
				}

				SR stepResults = servos->step(predictedActions);

				if (parameters->config->trainMode) {;
					
					for (int servo = 0; servo < NUM_SERVOS; servo++) {
						trainData[servo] = stepResults.servos[servo];
						trainData[servo].currentState = currentState[servo];
						currentState[servo] = trainData[servo].nextState;

						if (replayBuffer->size() <= parameters->config->maxBufferSize) {
							replayBuffer->add(trainData[servo]);
						}
					}

					if (!trainData[1].done) {
						episodeSteps += 1.0;
						episodeRewards += trainData[1].reward;
					}
					else {
						numEpisodes += 1;


						// EMA of steps and rewards (With 30% weight to new episodes; or 5 episode averaging)
						double percentage = (1.0 / 3.0);
						double timePeriods = (2.0 / percentage) - 1.0;
						double emaWeight = (2.0 / (timePeriods + 1.0));

						episodeAverageRewards = (episodeRewards - episodeAverageRewards) * emaWeight + episodeAverageRewards;
						episodeAverageSteps = (episodeSteps - episodeAverageSteps) * emaWeight + episodeAverageSteps;

						// Log Episode averages
						std::string avePath = get_current_dir_name() + statFileName;
						std::string episodeData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
												 (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
												+ std::to_string(numEpisodes) + ','
												+ std::to_string(episodeRewards) + ','
												+ std::to_string(episodeSteps) + ','
												+ std::to_string(episodeAverageRewards) + ','
												+ std::to_string(episodeAverageSteps);
						appendLineToFile(avePath, episodeData);

						episodeSteps = 0.0;
						episodeRewards = 0.0;
					}

		
                    if (replayBuffer->size() > parameters->config->minBufferSize + 1) {
                        kill(parameters->pid, SIGUSR1);
                    } 
				}
			}
			else {
				resetResults = servos->reset();
				for (int servo = 0; servo < NUM_SERVOS; servo++) {
					currentState[servo] = resetResults.servos[servo];
				}		
			}
		}
		else {
			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {
					predictedActions[i][0] = parameters->config->defaultGains[0];
					predictedActions[i][1] = parameters->config->defaultGains[1];
					predictedActions[i][2] = parameters->config->defaultGains[2];
				}

				servos->step(predictedActions, false);

				// TODO remove this after testing
				if (replayBuffer->size() <= parameters->config->maxBufferSize) {
					TD data;
					replayBuffer->add(data);
				}
				if (replayBuffer->size() > parameters->config->minBufferSize + 1) {
					kill(parameters->pid, SIGUSR1);
				} 
			}
			else {
				resetResults = servos->reset();
			}	
		}
	}
}

void* detectThread(void* args)
{
	using namespace TATS;
	using namespace Utility;
	param* parameters = (param*)args;
	int fd = parameters->fd;

	// user hyperparams
	float recheckChance = parameters->config->recheckChance;
	int trackerType = parameters->config->trackerType;
	bool useTracking = parameters->config->useTracking;
	bool draw = parameters->config->draw;
	bool showVideo = parameters->config->showVideo;
	bool cascadeDetector = parameters->config->cascadeDetector;

	// program state variables
	bool rechecked = false;
	bool isTracking = false;
	bool isSearching = false;
	int lossCount = 0;
	int lossCountMax = parameters->config->lossCountMax;

	// Create object tracker to optimize detection performance
	cv::Rect2d roi;
	cv::Ptr<cv::Tracker> tracker;
	if (useTracking) {
		tracker = createOpenCVTracker(trackerType);
	}

	// Object center coordinates
	double frameCenterX = 0;
	double frameCenterY = 0;

	// Object coordinates
	double objX = 0;
	double objY = 0;

	cv::Mat frame;
	cv::Mat detection;
	std::chrono::steady_clock::time_point Tbegin, Tend;
	auto execbegin = std::chrono::high_resolution_clock::now();

	if (!camera.isOpened())
	{
		throw std::runtime_error("cannot initialize camera");
	}

	std::string path = get_current_dir_name();
	Detect::CascadeDetector cd("./models/haar/haarcascade_frontalface_alt2.xml");
	std::vector<struct Detect::DetectionData> results;

	while (true) {

		if (isSearching) {
			// TODO:: Perform better search ruetine
			// For now servo thread detects when done and sends a reset commman to servos
		}

		try
		{
			frame = GetImageFromCamera(camera);

			if (frame.empty())
			{
				continue;
			}

			if (!useTracking) {
				goto detect;
			}

			if (isTracking) {
				isSearching = false;

				// Get the new tracking result
				if (!tracker->update(frame, roi)) {
					isTracking = false;
					lossCount++;
					goto detect;
				}

				// Chance to revalidate object tracking quality
				if (recheckChance >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
					rechecked = true;
					goto detect;
				}

			validated:
				ED tilt;
				ED pan;

				// Determine object and frame centers
				frameCenterX = static_cast<double>(frame.cols) / 2.0;
				frameCenterY = static_cast<double>(frame.rows) / 2.0;
				objX = roi.x + roi.width * 0.5;
				objY = roi.y + roi.height * 0.5;

				// Determine error
				tilt.error = frameCenterY - objY;
				pan.error = frameCenterX - objX;

				// Enter State data
				pan.point = roi.x;
				tilt.point = roi.y;
				pan.size = roi.width;
				tilt.size = roi.height;
				pan.obj = objX;
				tilt.obj = objY;
				pan.frame = frameCenterX;
				tilt.frame = frameCenterY;
				pan.done = false;
				tilt.done = false;
				
				double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
				pan.timestamp = elapsed;
				tilt.timestamp = elapsed;

				// Fresh data
                ED eventDataArray[2] {
                    tilt,
					pan
                };
                
                servos->update(eventDataArray);

				// draw to frame
				if (draw) {
					// TODO:: Draw bb on frame
				}
			}
			else {

			detect:
				results = cd.detect(frame, draw, 1);
                Detect::DetectionData result = results.at(0);

				if (result.found) {

					// Update loop variants
					lossCount = 0;
					isSearching = false;

					if (rechecked) {
						rechecked = false;
						goto validated; 
					}

					ED tilt;
					ED pan;

					// Determine object and frame centers
					frameCenterX = static_cast<double>(frame.cols) / 2.0;
					frameCenterY = static_cast<double>(frame.rows) / 2.0;
					objX = static_cast<double>(result.center.x);
					objY = static_cast<double>(result.center.y);

					// Determine error (negative is too far left or too far above)
					tilt.error = frameCenterY - objY;
					pan.error = frameCenterX - objX;

					// Other State data
					double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
					pan.timestamp = elapsed;
					tilt.timestamp = elapsed;

					pan.point = static_cast<double>(result.boundingBox.x);
					tilt.point = static_cast<double>(result.boundingBox.y);
					pan.size = static_cast<double>(result.boundingBox.width);
					tilt.size = static_cast<double>(result.boundingBox.height);
					pan.obj = objX;
					tilt.obj = objY;
					pan.frame = frameCenterX;
					tilt.frame = frameCenterY;
					pan.done = false;
					tilt.done = false;

					// Fresh data
                    ED eventDataArray[2] {
                        tilt,
                        pan
                    };
                    
                    servos->update(eventDataArray);

					if (useTracking) {

						roi.x = result.boundingBox.x;
						roi.y = result.boundingBox.y;
						roi.width = result.boundingBox.width;
						roi.height = result.boundingBox.height;

						tracker = createOpenCVTracker(trackerType);
						if (tracker->init(frame, roi)) {
							isTracking = true;
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

						ED tilt;
						ED pan;
						
						// Object not on screen
						frameCenterX = static_cast<double>(frame.cols) / 2.0;
						frameCenterY = static_cast<double>(frame.rows) / 2.0;
						objX = 0;
						objY = 0;

						// Max error
						tilt.error = frameCenterY;
						pan.error = frameCenterX;

						// Error state
						// Enter State data
						double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - execbegin).count() * 1e-9;
						pan.timestamp = elapsed;
						tilt.timestamp = elapsed;
						pan.point = 0;
						tilt.point = 0;
						pan.size = 0;
						tilt.size = 0;
						pan.obj = frameCenterX * 2;
						tilt.obj = frameCenterY * 2; // max error
						pan.frame = frameCenterX;
						tilt.frame = frameCenterY;
						pan.done = true;
						tilt.done = true;

						// Fresh data
                        ED eventDataArray[2] {
                            tilt,
                            pan
                        };
                        
                        servos->update(eventDataArray);
                    } 
				}
			}

			if (showVideo) {
				cv::imshow("Viewport", frame);
				cv::waitKey(1);
			}
		}
		catch (const std::exception&)
		{
			throw std::runtime_error("Issue detecting target from video");
		}
	}

	return NULL;
}

// void* autoTuneThread(void* args)
// {
// 	param* parameters = (param*)args;
// 	int batchSize = parameters->config->batchSize;
// 	long maxTrainingSteps = parameters->config->maxTrainingSteps;
// 	long currentSteps = 0;
// 	int minBufferSize = parameters->config->minBufferSize;
// 	int maxBufferSize = parameters->config->maxBufferSize;
// 	int sessions = parameters->config->maxTrainingSessions;
// 	int numUpdates = parameters->config->numUpdates;
// 	double rate = parameters->config->trainRate;

// 	// Annealed ERE (Emphasizing Recent Experience)
// 	// https://arxiv.org/pdf/1906.04009.pdf
// 	double N0 = 0.996;
// 	double NT = 1.0;
// 	double T = maxTrainingSteps;
// 	double t_i = 0;
	
// start:

// 	parameters->freshData = false;
// 	replayBuffer->clear();
	
// 	if (sessions <= 0) {
// 		parameters->config->trainMode = false; // start running in eval mode
// 		std::cout << "Training session over!!" << std::endl;
// 	}
	
// 	// Wait for the buffer to fill
// 	pthread_mutex_lock(&trainLock);
// 	while (!parameters->freshData) {
// 		pthread_cond_wait(&trainCond, &trainLock);
// 	}
// 	pthread_mutex_unlock(&trainLock);

// 	while (true) {

// 		double N = static_cast<double>(replayBuffer->size());
// 		t_i += 1;
// 		double n_i = N0 + (NT - N0) * (t_i / T);
// 		int cmin = N - ( minBufferSize );
		
		
//         if (currentSteps >= maxTrainingSteps) {
//             currentSteps = 0;
//             sessions--;
//             t_i = 0;
//             goto start;
//         }
//         else {
//             currentSteps += 1;
//         }

//         for (int k = 0; k < numUpdates; k += 1) {
//             int startingRange = std::min<int>( N - N * std::pow(n_i, static_cast<double>(k) * (1000.0 / numUpdates)), cmin);

//             TrainBuffer batch = replayBuffer->ere_sample(batchSize, startingRange);
//             pidAutoTuner->update(batch.size(), &batch);
//         }

//         long milis = static_cast<long>(1000.0 / rate);
//         msleep(milis);
// 	}

// 	return NULL;
// }


// g++ -o test /home/zachoines/Documents/repos/test/pytorch/test.cpp -std=gnu++17 -Wl,--no-as-needed -g -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
// cmake -DCMAKE_PREFIX_PATH=/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/
// make TATS

/*
    torch::Device cpu(torch::kCPU);
    torch::Device cuda(torch::kCUDA);

    // Set up Tensorflow DNN
    std::string path = get_current_dir_name();
    std::string frozen_model_pbtxt = path + "/models/frozen_models/FacesMotorbikesairplanesModel.pbtxt";
	std::string frozen_model_pb = path + "/models/frozen_models/FacesMotorbikesairplanesModel.pb";
    std::vector<std::vector<cv::Mat>> outputblobs;
    cv::Mat display_image, input_image;
    cv::dnn::Net tensorflowDetector = cv::dnn::readNetFromTensorflow(frozen_model_pb);
    tensorflowDetector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    tensorflowDetector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // One time initialization of DNN
    std::string imageFile = "/images/faces/image_0001.jpg";
    std::string imageFilePath = path + imageFile;
    input_image = cv::imread(imageFilePath, cv::IMREAD_UNCHANGED);
    input_image.convertTo(input_image, CV_32F, 1 / 255.0);
    tensorflowDetector.setInput(cv::dnn::blobFromImage(input_image, 1.0, cv::Size(224, 224), 0.0, false, false, CV_32F));
    tensorflowDetector.forward(outputblobs, {"functional_1/box_output/Sigmoid", "functional_1/label_output/Softmax"}); 

    outputblobs.clear();
    input_image = GetImageFromCamera(camera);
    display_image = input_image.clone();
    input_image.convertTo(input_image, CV_32F, 1 / 255.0);
    cv::resize(input_image, input_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    tensorflowDetector.setInput(cv::dnn::blobFromImage(input_image, 1.0, cv::Size(224, 224), 0.0, true, false, CV_32F));
    tensorflowDetector.forward(outputblobs, {"functional_1/box_output/Sigmoid", "functional_1/label_output/Softmax"});
    cv::Mat box = outputblobs.at(0).at(0);
    cv::Mat probs = outputblobs.at(1).at(0);
    
    int h = display_image.rows;
    int w = display_image.cols;

    // Print what we see
    std::cout << box << std::endl;
    std::cout << probs << std::endl;
    
    // Box Image dims
    double startYProb = box.at<float>(0);
    double endYProb = box.at<float>(1);
    double startXProb = box.at<float>(2);
    double endXProb = box.at<float>(3);
    
    int startX = static_cast<int>(startXProb * static_cast<double>(w));
    int startY = static_cast<int>(startYProb * static_cast<double>(h));
    int endX = static_cast<int>(endXProb * static_cast<double>(w));
    int endY =  static_cast<int>(endYProb * static_cast<double>(h));

    // Argmax of probs
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    cv::minMaxLoc(probs, &minVal, &maxVal, &minLoc, &maxLoc );
    drawPred(maxVal, startX, startY, endX, endY, display_image, "test");
    cv::imshow("CSI Camera", display_image);
*/



