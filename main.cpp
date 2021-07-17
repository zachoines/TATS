// C Libs
#include <signal.h>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// Sensor libs
#include "RF24/RF24.h"

// Opencv inpots
#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"

// Local imports
#include "./src/TATS/TATS.h"

// Local Function hoisting
unsigned long getTimestamp(struct Signal& s);
int mapSpeeds(int val);
void radioThread(Utility::param* parameters);
void trackingThread(Utility::param* parameters);

// Signal handlers
static void usr_sig_handler1(const int sig_number, siginfo_t* sig_info, void* context);
volatile sig_atomic_t sig_value1;

// Data for RF24
union bytesToTimestamp {
  unsigned char buffer[4];
  unsigned long timestamp;
} timestamp_converter;
enum data { wLeft = 0, wRight = 1, sUp = 2, sDown = 3, t1 = 4, t2 = 5, t3 = 6, t4 = 7 };
struct Signal {
  unsigned char buffer[8];

  bool operator == (struct Signal& a)
  {
    for (int i = 0; i < 4; i++) {
      if (abs(buffer[i] - a.buffer[i]) > 2) {
        return false;
      }
    }
    
    return true;
  }

  struct Signal& operator = (struct Signal& a)
  {
    for (int i = 0; i < 8; i++) {
      this->buffer[i] = a.buffer[i];
    }

    return *this;
  }
};

// Gobal variables for threads
cv::VideoCapture* camera = nullptr;
RF24* radio = nullptr;
TATS::Config* config = nullptr
Utility::param* parameters = nullptr;

int main(int argc, char** argv)
{
    using namespace Utility;
    using namespace TATS;
    using namespace cv;
    using namespace std;

    srand( (unsigned)time( NULL ) );

    // Other loops variables
    pid_t pid = -1;
    Signal oldData;
    Signal newData; 

    config = new Config();
    TATS targetTrackingSystem(&config);
    
    // Parent process is image recognition PID/servo controller, second is SAC PID Autotuner
    if (config->multiProcess && config->trainMode) {
        pid = fork(); 
    } else {
        pid = getpid();
    }

    if (pid > 0) {
        // initialize as a parent TATS instance
        targetTrackingSystem.init(pid);
        parameters->pid = pid;
        std::thread radioThread(radioThread, parameters);
        std::thread cameraThread(trackingThread, parameters);

        if (config.trainMode) {
            radioThread.detach();
            trackingThread.detach();
            targetTrackingSystem.syncThread(parameters); // Blocking call
        } else {
            radioThread.join();
            trackingThread.join();
        }  

        // Terminate and wait for child processes on exit
        kill(pid, SIGQUIT);
        if (wait(NULL) != -1) {
            return 0;
        }
        else {
            return -1;
        }

    } else { // Only if in training mode for TATS
        // Kill child if parent killed
        prctl(PR_SET_PDEATHSIG, SIGKILL); 
        parameters->pid = pid;        

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

        // initialize as a child TATS instance
        targetTrackingSystem.init(pid);

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
                while (!targetTrackingSystem.trainTATSChildProcess()) {
        
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

int mapSpeeds(int val) {
  int adjusted;
  int max = 255;
  int middle = max / 2;
  int clipped = std::clamp(val, 0, max);

  // Not one-to-one mapping of microseconds on lower and upper speed ranges
  if (clipped < middle) {
    adjusted = Utility::mapOutput(clipped, 0, middle, 455, 1550); 
  } else {
    adjusted = Utility::mapOutput(clipped, middle, max, 1550, 2250);
  }

  return adjusted;
}

unsigned long getTimestamp(struct Signal& s) {

  timestamp_converter.buffer[0] = s.buffer[data::t1];
  timestamp_converter.buffer[1] = s.buffer[data::t2];
  timestamp_converter.buffer[2] = s.buffer[data::t3];
  timestamp_converter.buffer[3] = s.buffer[data::t4];

  return timestamp_converter.timestamp;
}

void trackingThread(Utility::param* parameters) {
    camera = new VideoCapture(0, CAP_GSTREAMER);
    camera->set(CAP_PROP_FRAME_WIDTH, config->captureSize[1]);
    camera->set(CAP_PROP_FRAME_HEIGHT, config->captureSize[0]);

    // Check Camera
    if (!camera->isOpened()) {
        throw std::runtime_error("cannot initialize camera");
    } else {
        if (GetImageFromCamera(camera).empty())
        {
            throw std::runtime_error("Issue reading frame!");
        }
    }

    // Register TATS event callbacks and initialize
    targetTrackingSystem.init(parameters->pid);

    while(true) {
        cv::Mat image = GetImageFromCamera(camera);
        targetTrackingSystem.update(&image);

        if (showVideo) {
            cv::imshow("Viewport", frame);
            cv::waitKey(1);
        }
    }
}

void radioThread(Utility::param* parameters) {
    
    // RF24 inits
    uint8_t address[] = { 0xE6, 0xE6, 0xE6, 0xE6, 0xE6 };
    uint8_t pipe = 0;
    uint16_t spiBus = 1;
    uint16_t CE_GPIO = 481;
    RF24 radio = new RF24(CE_GPIO, spiBus); // SPI Bus 1 (0 or 1) and GPIO17_40HEADER (pin 22)
    
    // Check radio
    if (!radio.begin(CE_GPIO, spiBus)) {
        std::cout << "radio hardware is not responding!!" << std::endl;
        throw std::runtime_error("Radio failde to initialize!");
    } else {
        radio.openReadingPipe(pipe, address);
        radio.startListening(); // Set to radio receiver mode
    }

    // Main program loop
    while(true) {
        if ( radio.available(&pipe) ) {
            oldData = newData;
            radio.read(&newData.buffer, 8);

            if (getTimestamp(newData) != getTimestamp(oldData)) {
                std::cout << "We have a match!!" << std::endl;
            }

            std::cout << std::to_string(mapSpeeds(newData.buffer[data::wLeft])) << std::endl;
            std::cout << std::to_string(mapSpeeds(newData.buffer[data::wRight])) << std::endl;
        } 
    }        
}