#include "./../src/tracking/TATS.h"
#include <RF24/RF24.h> 
#include <atomic>
#include <csignal>
#include <sys/prctl.h>  /* prctl */

// Local Function hoisting
double mapSpeed(unsigned char val, double map_low, double map_mid, double map_high);

// Signal handlers
static void sig_handler(int signum);
std::atomic<bool> stopFlag = false;

// Data for RF24
enum data { LeftY = 0, LeftX = 1, RightY = 2, RightX = 3, L1 = 4, R1 = 5, t1 = 6, t2 = 7, t3 = 8, t4 = 9};
struct Signal {
  unsigned char buffer[10] = { 0 };

  bool operator == (struct Signal& a)
  {
    for (int i = 0; i < 6; i++) {
      if (abs(buffer[i] - a.buffer[i]) > 0) {
        return false;
      }
    }
    
    return true;
  }

  struct Signal& operator = (struct Signal& a)
  {
    for (int i = 0; i < 10; i++) {
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
double previousActions[2] = { 0.0 };
double speeds[2] = { 0.0 };
double speed = .20; // For Hitech D951TW (.14sec per 60 deg)
double anglesPerSecondScaled = 60.0 / speed;
double maxDeltaAngle = 15.0;
std::atomic<bool> servosOverride = false;
std::chrono::steady_clock::time_point start;

// Overridden with custom logic, alternative to the callbacks if only one TATS object
void control::TATS::onTargetUpdate(control::INFO info, EVENT eventType) {
    switch (eventType) {
        case EVENT::ON_UPDATE:
            if (info.tracking) {
                // std::cout << "Currently tracked target: " << config->classes[info.id] << std::endl;
            }
            return;
        default:
            return;
    }
}

bool control::TATS::actionOverride(double actions[2][1], Utility::ActionType actionType) {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    double dt = std::clamp<double>(std::chrono::duration<double>(now - start).count(), 0.0, 1.0); // In seconds
    start = now;
  
    if (servosOverride) {
        switch(actionType) {
            case Utility::ActionType::SPEED:
                actions[1][0] = speeds[1];
                actions[0][0] = speeds[0];
                return true;
            case Utility::ActionType::ANGLE:
                for (int i = 0; i < 2; i++) {
                    double preAct = Utility::mapOutput(previousActions[i], -1.0, 1.0, config->anglesLow[i], config->anglesHigh[i]);
                    actions[i][0] = std::clamp<double>(anglesPerSecondScaled * speeds[i] * dt, -maxDeltaAngle, maxDeltaAngle) + preAct;
                    actions[i][0] = Utility::mapOutput(std::clamp<double>(actions[i][0], config->anglesLow[i], config->anglesHigh[i]), config->anglesLow[i], config->anglesHigh[i], -1.0, 1.0);
                }
                return true;
        }
    } else {
        return false;
    }
    
}

void control::TATS::onServoUpdate(double pan, double tilt) {
    previousActions[1] = pan;
    previousActions[0] = tilt;
}

int main() {
    signal(SIGTSTP, sig_handler);  // CTRL-Z on terminal (kill -SIGTSTP PID)

    parameters = new Utility::Parameter();
    config = new Utility::Config();

    // Setup eval mode options
    config->trainMode = false; /* Disable train mode */
    config->logOutput = false; /* Disable debug messages */
    config->actionType = Utility::ActionType::SPEED;
    config->detector = Utility::DetectorType::CASCADE; /* Faster and more precise for training */
    config->detectorPath = "/models/haar/haarcascade_frontalface_default.xml"; 
    config->targets = { "face" };
    config->classes = { "face" };

    wire = new control::Wire();
    pwm = new control::PCA9685(0x40, wire);
    servos = new control::ServoKit(pwm);
    targetTrackingSystem = new control::TATS(config, servos);
   
    std::thread radioThreadT([&] {
        // RF24 inits
        uint8_t address[] = { 0xE6, 0xE6, 0xE6, 0xE6, 0xE6 };
        uint8_t pipe = 0;
        uint16_t spiBus = 0;
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
                radio.read(&newData.buffer, 10);
                
                if (newData.buffer[data::L1] || newData.buffer[data::R1]) {
                    servosOverride = !servosOverride;
                }

                if (!servosOverride) {
                    double wls = mapSpeed(newData.buffer[data::LeftY], 750.0, 1450.0, 2250.0);
                    double wrs = mapSpeed(newData.buffer[data::RightY], 750.0, 1450.0, 2250.0);
                    pwm->writeMicroseconds(10, wls);
                    pwm->writeMicroseconds(11, wrs);    
                } else {
                    speeds[0] = mapSpeed(newData.buffer[data::LeftY], -1.0, 0.0, 1.0);
                    speeds[1] = mapSpeed(newData.buffer[data::RightX], -1.0, 0.0, 1.0);
                }
            }
        }       
    });
        
    std::thread trackingThreadT([&] {
        targetTrackingSystem->init(1);
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
        }
    });

    radioThreadT.join();
    trackingThreadT.join();

    delete targetTrackingSystem;
    delete servos;
    delete pwm;
    delete wire;
    delete parameters;
    delete config;
}

static void sig_handler(int signum) {
    stopFlag = true;
}

double mapSpeed(unsigned char val, double map_low, double map_mid, double map_high) {

    double adjusted;
    double max = 255.0;
    double middle = 127.0;
    double value = static_cast<double>(val);

    // Not one-to-one mapping of microseconds on lower and upper speed ranges
    if (value <= middle) {
        adjusted = Utility::mapOutput(value, 0.0, middle, map_low, map_mid); 
    } else {
        adjusted = Utility::mapOutput(value, middle + 1.0, max, map_mid, map_high);
    }

    return adjusted;
}