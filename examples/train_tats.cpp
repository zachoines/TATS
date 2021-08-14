#include "./../src/tracking/TATS.h"
#include <RF24/RF24.h> 
#include <csignal>
#include <sys/prctl.h>  /* prctl */

// Signal handlers
static void sig_handler_child(const int sig_number, siginfo_t* sig_info, void* context);
static void sig_handler_parent(int signum);
volatile sig_atomic_t sig_value1;
volatile bool stopFlag = false;

Utility::param* parameters = nullptr;
Utility::Config* config = nullptr;
control::Wire* wire = nullptr;
control::PCA9685* pwm = nullptr;
control::ServoKit* servos = nullptr;
control::TATS* targetTrackingSystem = nullptr;


int main() {
    signal(SIGTSTP, sig_handler_parent);  // CTRL-Z on terminal (kill -SIGTSTP PID)
    srand( (unsigned)time( NULL ) );

    parameters = new Utility::Parameter();
    config = new Utility::Config();

    // Setup train mode options
    config->trainMode = true; /* Enable Train mode */
    config->initialRandomActions = false; /* Fill replay buffer with random experiances */
    config->stepsWithPretrainedModel = true; /* for transfer learning */
    config->lossCountMax = 0; /* Slows training down */
    config->multiProcess = true; /* Offloads SAC training in another process */
    config->disableServo[0] = true; /* Turn off tilt during training */
    config->disableServo[1] = false; /* Turn on pan during training */
    config->detector = Utility::DetectorType::CASCADE; /* Faster and more precise for training */
    config->detectorPath = "/models/haar/haarcascade_frontalface_default.xml"; 
    config->targets = { "face" };
    config->classes = { "face" };

    wire = new control::Wire();
    pwm = new control::PCA9685(0x40, wire);
    servos = new control::ServoKit(pwm);
    targetTrackingSystem = new control::TATS(config, servos);
    pid_t pid = -1;

    // Parent process is image recognition PID/servo controller, second is SAC PID Autotuner
    if (config->multiProcess && config->trainMode) {
        pid = fork(); 
    } else {
        pid = getpid();
    }

    if (pid > 0) {
        
        // initialize as a parent TATS instance
        std::thread trackingThreadT([&] {
            targetTrackingSystem->init(pid); 
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

        // 
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
                    throw std::runtime_error("sigwaitinfo() failed");
                }

                // Update weights on signal received from autotune thread/process
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
        // });
        
        trackingThreadT.join();
        // syncThread.join();

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

        std::cout << "Autotune Process: TATS initialized" << std::endl;

        // Wait on parent process' signal before training
        while (
            (sig_value1 != SIGINT) && 
            (sig_value1 != SIGTERM) && 
            (sig_value1 != SIGSEGV) && 
            (sig_value1 != SIGTSTP)) {
            
            sig_value1 = 0;

            // Sleep until signal is caught; train model on waking
            std::cout << "Autotune Process: Waiting for train signal" << std::endl;            
            sigsuspend(&zeromask);

            if (sig_value1 == SIGUSR1) {

                std::cout << "Autotune Process: Train signal received" << std::endl;
                while (!targetTrackingSystem->trainTATSChildProcess()) {
                    
                    // Inform parent new params are available
                    std::cout << "Autotune Process: Sending Sync signal" << std::endl;
                    kill(getppid(), SIGUSR1);

                    // Sleep per train rate
                    long milis = static_cast<long>(1000.0 / config->trainRate);
                    std::cout << "Autotune Process: Train session sucessful. Sleeping..." << std::endl;
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