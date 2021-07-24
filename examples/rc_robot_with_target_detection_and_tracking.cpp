#include "./../src/tracking/TATS.h"
#include <RF24/RF24.h> 
#include <csignal>
#include <sys/prctl.h>  /* prctl */

// Local Function hoisting
unsigned long getTimestamp(struct Signal& s);
int mapSpeeds(int val);

// Signal handlers
static void sig_handler_child(const int sig_number, siginfo_t* sig_info, void* context);
static void sig_handler_parent(int signum);
volatile sig_atomic_t sig_value1;
volatile bool stopFlag = false;

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

Utility::param* parameters = nullptr;
Utility::Config* config = nullptr;
control::Wire* wire = nullptr;
control::PCA9685* pwm = nullptr;
control::ServoKit* servos = nullptr;
control::TATS* targetTrackingSystem = nullptr;

int main() {
    signal(SIGTSTP, sig_handler_parent);  // CTRL-Z on terminal (kill -SIGTSTP PID)
    srand( (unsigned)time( NULL ) );

    // Pointers here
    parameters = new Utility::Parameter();
    config = new Utility::Config();
    wire = new control::Wire();
    pwm = new control::PCA9685(0x40, wire);
    servos = new control::ServoKit(pwm);
    targetTrackingSystem = new control::TATS(*config, servos);
    pid_t pid = -1;

    // Parent process is image recognition PID/servo controller, second is SAC PID Autotuner
    if (config->multiProcess && config->trainMode) {
        pid = fork(); 
    } else {
        pid = getpid();
    }

    if (pid > 0) {
        
        // Register event callbacks
        std::function<void(control::INFO const&)> searchCallback = [](control::INFO event) {
            // TODO:: Logic that moves motors and servos in a search pattern
            pwm->writeMicroseconds(2, 1500); // for example reset motors
            pwm->writeMicroseconds(3, 1500);
        };

        std::function<void(control::INFO const&)> updateCallback = [](control::INFO event) {
            // TODO:: Logic that moves car towards the target given angle and distance (If seperate lidar sensor is attached to servos)
        };

        std::function<void(control::INFO const&)> lossCallback = [](control::INFO event) {
            // TODO:: Logic that precedes a search routine
            pwm->writeMicroseconds(2, 1500); // for example reset motors
            pwm->writeMicroseconds(3, 1500);
        };

        targetTrackingSystem->registerCallback(control::EVENT::ON_SEARCH, searchCallback);
        targetTrackingSystem->registerCallback(control::EVENT::ON_UPDATE, updateCallback);
        targetTrackingSystem->registerCallback(control::EVENT::ON_LOST, lossCallback);
        
        // initialize as a parent TATS instance
        targetTrackingSystem->init(pid);     

        std::thread radioThreadT([&] {
            // RF24 inits
            uint8_t address[] = { 0xE6, 0xE6, 0xE6, 0xE6, 0xE6 };
            uint8_t pipe = 0;
            uint16_t spiBus = 1;
            uint16_t CE_GPIO = 481;
            RF24 radio(CE_GPIO, spiBus); // SPI Bus 1 (0 or 1) and GPIO17_40HEADER (pin 22)

            Signal oldData;
            Signal newData; 
            
            // Check radio
            if (!radio.begin(CE_GPIO, spiBus)) {
                std::cout << "radio hardware is not responding!!" << std::endl;
                throw std::runtime_error("Radio faild to initialize!");
            } else {
                radio.openReadingPipe(pipe, address);
                radio.startListening(); // Set to radio receiver mode
            }

            // Main program loop
            while(!stopFlag) {
                if ( radio.available(&pipe) ) {
                    oldData = newData;
                    radio.read(&newData.buffer, 8);

                    if (getTimestamp(newData) != getTimestamp(oldData)) {
                        double wls = mapSpeeds(newData.buffer[data::wLeft]);
                        double wrs = mapSpeeds(newData.buffer[data::wRight]);
                        pwm->writeMicroseconds(0, wls);
                        pwm->writeMicroseconds(1, wrs);
                    }
                } 
            }       
        });
        
        std::thread trackingThreadT([&] {
            cv::VideoCapture* camera = new cv::VideoCapture(0, cv::CAP_GSTREAMER);
            camera->set(cv::CAP_PROP_FRAME_WIDTH, config->captureSize[1]);
            camera->set(cv::CAP_PROP_FRAME_HEIGHT, config->captureSize[0]);

            // Check Camera
            if (!camera->isOpened()) {
                throw std::runtime_error("cannot initialize camera");
            } else {
                if (Utility::GetImageFromCamera(camera).empty())
                {
                    throw std::runtime_error("Issue reading frame!");
                }
            }

            while(!stopFlag) {
                cv::Mat image = Utility::GetImageFromCamera(camera);
                targetTrackingSystem->update(image);

                if (config->showVideo) {
                    cv::imshow("Viewport", image);
                    cv::waitKey(1);
                }
            }
        });

        std::thread syncThread([&] {
            if (config->trainMode) {

                // The below loops is the sync for multiprocess training of TATS
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
                while (!stopFlag) {
                    
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
                            targetTrackingSystem->syncTATS();
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
        });
        
        radioThreadT.join();
        trackingThreadT.join();
        syncThread.join();

        delete targetTrackingSystem;
        delete servos;
        delete pwm;
        delete wire;
        delete parameters;
        delete config;
    
        // Terminate and wait for child processes before exit
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
        sig_action.sa_sigaction = sig_handler_child;

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
        targetTrackingSystem->init(pid);

        // Wait on parent process' signal before training
        while (
            (sig_value1 != SIGINT) && 
            (sig_value1 != SIGTERM) && 
            (sig_value1 != SIGSEGV) && 
            (sig_value1 != SIGTSTP)) {
            
            sig_value1 = 0;

            // Sleep until signal is caught; train model on wakin            
            sigsuspend(&zeromask);

            if (sig_value1 == SIGUSR1) {

                std::cout << "Train signal received..." << std::endl;

                // Begin training process
                while (!targetTrackingSystem->trainTATSChildProcess()) {
        
                    // Inform parent new params are available
                    kill(getppid(), SIGUSR1);

                    // Sleep per train rate
                    long milis = static_cast<long>(1000.0 / config->trainRate);
                    Utility::msleep(milis);
                }

                return 0;
            }
        }
    }
}

static void sig_handler_child(const int sig_number, siginfo_t* sig_info, void* context){
    
    // Take care of all segfaults
    if (sig_number == SIGSEGV || sig_number == SIGINT || sig_number == SIGSTOP)
    {
        kill(getpid(), SIGKILL);  
    } 

    sig_value1 = sig_number;
}

static void sig_handler_parent(int signum) {
    stopFlag = true;
}

int mapSpeeds(int val) {
  int adjusted;
  int max = 255;
  int middle = max / 2;
  int clipped = std::clamp(val, 0, max);

  // Not one-to-one mapping of microseconds on lower and upper speed ranges
  if (clipped < middle) {
    adjusted = Utility::mapOutput(clipped, 0, middle, 490, 1550); 
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