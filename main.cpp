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
void panTiltThread(Utility::param* parameters);
void detectThread(Utility::param* parameters);
void syncThread(Utility::param* parameters);

// Thread Sync
pthread_mutex_t stateDataLock = PTHREAD_MUTEX_INITIALIZER;
pid_t pid;

// Shared between threads
SACAgent* pidAutoTuner = nullptr;
TATS::Env* servos = nullptr;
cv::VideoCapture* camera = nullptr;

// Log files
std::string statFileName = "/stat/episodeAverages.txt";
std::string lossFileName = "/stat/trainingLoss.txt";
std::string stateDir = "/stat";

// Signal handlers
static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
volatile sig_atomic_t sig_value1;


int main(int argc, char** argv)
{

	// Kill child if parent dies
	prctl(PR_SET_PDEATHSIG, SIGKILL); 

	using namespace Utility;
	using namespace TATS;

    // Initialize defaults
    param* parameters = new Parameter();
	Config* config = new Config();
    servos = new TATS::Env();

	// Init camera
	// width=3264, height=2464
	//std::string pipeline = "nvarguscamerasrc sensor-id=1 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 wbmode=3 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=120/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.3 brightness=-.2 saturation=1.2 ! appsink";

	// std::string pipeline = gstreamer_pipeline(0, config->dims[1], config->dims[0], config->dims[1], config->dims[0], config->maxFrameRate, 0);
	std::string pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=4032,height=3040,framerate=30/1 ! nvvidconv ! video/x-raw, width=1280,height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.3 brightness=-.2 saturation=1.2 ! appsink";
	camera = new cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);

	// Shared memory for SAC model syncing
	int ShmID = shmget(IPC_PRIVATE, 10000 * sizeof(double), IPC_CREAT | 0666);
	if (ShmID < 0) {
        throw std::runtime_error("Could not initialize shared memory");
	}

	parameters->ShmID = ShmID;

	// Create a shared memory buffer for experiance replay
	boost::interprocess::shared_memory_object::remove("SharedMemorySegment");
	boost::interprocess::managed_shared_memory segment(boost::interprocess::create_only, "SharedMemorySegment", sizeof(TD) * config->maxBufferSize + 1);
	const ShmemAllocator alloc_inst(segment.get_segment_manager());
	SharedBuffer* sharedTrainingBuffer = segment.construct<SharedBuffer>("SharedBuffer") (alloc_inst);

    // Setup stats files
    std::string avePath = get_current_dir_name() + statFileName;
	std::string lossPath = get_current_dir_name() + lossFileName;
	std::string statPath = get_current_dir_name() + stateDir;

	mkdir(statPath.c_str(), 0755);

	// Remove old logs
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

    // Setup pids
	parameters->isTraining = false;
	parameters->freshData = false;

    // Parent process is image recognition PID/servo controller, second is SAC Servo autotuner
    pid = fork();
	if (pid > 0) {

        parameters->pid = pid;

        // Setup threads and PIDS
        pidAutoTuner = new SACAgent(config->numInput, config->numHidden, config->numActions, config->actionHigh, config->actionLow);

		// std::thread syncT(syncThread, parameters);
		std::thread panTiltT(panTiltThread, parameters);
		std::thread detectT(detectThread, parameters);

		panTiltT.detach();
		detectT.detach();
		syncThread(parameters);

		// Terminate Child processes
		kill(-pid, SIGQUIT);
		if (wait(NULL) != -1) {
			return 0;
		}
		else {
			return -1;
		}

    } else {
        
		SACAgent* pidAutoTunerchild = new SACAgent(config->numInput, config->numHidden, config->numActions, config->actionHigh, config->actionLow);

        // Get Shared memory reference for the child
		double* ShmPTRChild;
		ShmPTRChild = (double*)shmat(ShmID, 0, 0);

		if (ShmPTRChild == nullptr) {
			throw std::runtime_error("Could not initialize shared memory");
		}

        // Retrieve the training buffer from shared memory
		boost::interprocess::managed_shared_memory _segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
		SharedBuffer* trainingBuffer = _segment.find<SharedBuffer>("SharedBuffer").first;
		ReplayBuffer* replayBuffer = new ReplayBuffer(config->maxBufferSize, trainingBuffer);

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

                    // TrainBuffer batch = replayBuffer->sample(batchSize);
                    // pidAutoTunerChild->update(batchSize, &batch);
					// int valuesWritten = pidAutoTunerChild->sync(false, ShmPTRChild);
					// std::cout << "Tensors written in child: " << valuesWritten << std::endl;

					replayBuffer->clear();
					Utility::msleep(2000);
					std::cout << "Sync signal sent..." << std::endl;
					kill(getppid(), SIGUSR1);
					isTraining = false;
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

void syncThread(Utility::param* parameters) {
	using namespace Utility;
	Utility::Config* config = new Utility::Config();
	
	// SHared memory referance for the parent process
	double* ShmPTRParent = (double*)shmat(parameters->ShmID, 0, 0);

	if (ShmPTRParent == nullptr) {
		throw std::runtime_error("Could not initialize shared memory");
	}

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

		signum = sigwaitinfo(&mask, &info);
		if (signum == -1) {
			if (errno == EINTR)
				continue;
			std::runtime_error("Parent process: sigwaitinfo() failed");
		}

		// Update weights on signal received from autotune thread
		if (signum == SIGUSR1 && info.si_pid == pid) {
			std::cout << "Received sync signal..." << std::endl;
			if (pthread_mutex_lock(&stateDataLock) == 0) {
				// int valuesRead = pidAutoTuner->sync(true, parameters->ShmPTR);
				// std::cout << "Tensors read in parent: " << valuesRead << std::endl;
			}

			pthread_mutex_unlock(&stateDataLock);
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
	if (sig_number == SIGSEGV)
	{
		perror("SIGSEV: Address access error.");
		exit(-1);
	}

	sig_value1 = sig_number;
}

void panTiltThread(Utility::param* parameters) {
	using namespace TATS;
	using namespace Utility;
	std::mt19937 eng{ std::random_device{}() };
    torch::Device device(torch::kCUDA);
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device);
	Utility::Config* config = new Utility::Config();

    // Retrieve the training buffer from shared memory
	boost::interprocess::managed_shared_memory _segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "SharedMemorySegment");
	SharedBuffer* trainingBuffer = _segment.find<SharedBuffer>("SharedBuffer").first;
	ReplayBuffer* replayBuffer = new ReplayBuffer(config->maxBufferSize, trainingBuffer);

	// Training options and record keeping
	double episodeAverageRewards = 0.0;
	double episodeAverageSteps = 0.0;
	int numEpisodes = 0;
	double episodeRewards = 0.0;
	double episodeSteps = 0.0;

	// training state variables
	bool initialRandomActions = config->initialRandomActions;
	int numInitialRandomActions = config->numInitialRandomActions;
	
	double predictedActions[NUM_SERVOS][NUM_ACTIONS];
	double stateArray[NUM_INPUT];
	SD currentState[NUM_SERVOS];
	TD trainData[NUM_SERVOS];
	ED eventData[NUM_SERVOS];
	RD resetResults;

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
				for (int i = 0; i < 2; i++) {

					// Query network and get PID gains
					if (config->trainMode) {
						if (initialRandomActions && numInitialRandomActions >= 0) {

							numInitialRandomActions--;
							std::cout << "Random action count: " << numInitialRandomActions << std::endl;
							for (int a = 0; a < config->numActions; a++) {
								predictedActions[i][a] = std::uniform_real_distribution<double>{ config->actionLow, config->actionHigh }(eng);;
							}
						}
						else {
							currentState[i].getStateArray(stateArray);
							at::Tensor actions = pidAutoTuner->get_action(torch::from_blob(stateArray, { 1, config->numInput }, options), true);

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

				SR stepResults;

				try {
					stepResults = servos->step(predictedActions);
				} catch (...) {
					throw std::runtime_error("cannot step with servos");
				}

				if (config->trainMode) {;
					
					for (int servo = 0; servo < NUM_SERVOS; servo++) {
						trainData[servo] = stepResults.servos[servo];
						trainData[servo].currentState = currentState[servo];
						currentState[servo] = trainData[servo].nextState;

						if (replayBuffer->size() <= config->maxBufferSize) {
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

		
                    if (replayBuffer->size() > config->minBufferSize + 1) {
                        kill(parameters->pid, SIGUSR1);
                    } 
				}
			}
			else {
				
				try {
					resetResults = servos->reset();
				} catch (...) {
					throw std::runtime_error("cannot reset servos");
				}
				
				
				for (int servo = 0; servo < NUM_SERVOS; servo++) {
					currentState[servo] = resetResults.servos[servo];
				}		
			}
		}
		else {
			if (!servos->isDone()) {
				for (int i = 0; i < 2; i++) {
					predictedActions[i][0] = config->defaultGains[0];
					predictedActions[i][1] = config->defaultGains[1];
					predictedActions[i][2] = config->defaultGains[2];
				}

				try {
					servos->step(predictedActions, false);
				} catch (...) {
					throw std::runtime_error("cannot step with servos");
				}

				// TODO remove this after testing
				if (replayBuffer->size() <= config->maxBufferSize) {
					TD data;
					replayBuffer->add(data);
				}
				if (replayBuffer->size() > config->minBufferSize + 1) {
					std::cout << "Sending train signal..." << std::endl;
					kill(parameters->pid, SIGUSR1);
				} 
			}
			else {

				try {
					resetResults = servos->reset();
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
	bool cascadeDetector = config->cascadeDetector;

	// program state variables
	bool rechecked = false;
	bool isTracking = false;
	bool isSearching = false;
	int lossCount = 0;
	int lossCountMax = config->lossCountMax;

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

	if (!camera->isOpened())
	{
		throw std::runtime_error("cannot initialize camera");
	}

	/*
	std::string path = get_current_dir_name();
	Detect::CascadeDetector detector("./models/haar/haarcascade_frontalface_alt2.xml");
	*/

	// load the network
	std::vector<std::string> class_names = {
		"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"
	};

	std::string path = get_current_dir_name();
	std::string weights = path + "/models/yolo/yolo_uno.torchscript.pt";
	Detect::YoloDetector detector = Detect::YoloDetector(weights, class_names);

	std::vector<struct Detect::DetectionData> results;

	while (true) {

		if (isSearching) {
			// TODO:: Perform better search ruetine
			// For now servo thread detects when done and sends a reset commman to servos
		}

		try
		{
			try {
				frame = GetImageFromCamera(camera);
			} catch (const std::exception& e)
			{
				std::cerr << e.what();
				camera->release();
				throw std::runtime_error("could not get image from camera");
			}

			if (frame.empty())
			{
				continue;
			}

			if (!useTracking) {
				goto detect;
			}

			if (isTracking) {
				isSearching = false;

				try {
					// Get the new tracking result
					if (!tracker->update(frame, roi)) {
						isTracking = false;
						lossCount++;
						goto detect;
					}
				} catch (const std::exception& e)
				{
					std::cerr << e.what();
					camera->release();
					throw std::runtime_error("could not update opencv tracker");
				}			

				// Chance to revalidate object tracking quality
				if (frameCount >= recheckFrequency) {
					frameCount = 0;
					rechecked = true;
					goto detect;
				} else {
					frameCount++;
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
					std::vector<struct Detect::DetectionData> out = detector.detect(frame, draw, 1);
					results = out;
				} catch (const std::exception& e)
				{
					std::cerr << e.what();
					camera->release();
					throw std::runtime_error("Could not detect target from frame");
				}
				
				Detect::DetectionData result;
				result.found = false;

				if (!results.empty()) { 
					result = results.at(0);
				} 
                 
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
                    
                    try {
						servos->update(eventDataArray);
					} catch (...) {
						throw std::runtime_error("cannot update event data");
					}

					if (useTracking) {

						try {
							roi = result.boundingBox;
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
			std::cerr << e.what();
			camera->release();
			throw std::runtime_error("Issue detecting target from video");
		}
	}
}

// void* autoTuneThread(void* args)
// {
// 	param* parameters = (param*)args;
// 	int batchSize = config->batchSize;
// 	long maxTrainingSteps = config->maxTrainingSteps;
// 	long currentSteps = 0;
// 	int minBufferSize = config->minBufferSize;
// 	int maxBufferSize = config->maxBufferSize;
// 	int sessions = config->maxTrainingSessions;
// 	int numUpdates = config->numUpdates;
// 	double rate = config->trainRate;

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
// 		config->trainMode = false; // start running in eval mode
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



