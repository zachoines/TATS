#include "./../src/tracking/TATS.h"
#include <RF24/RF24.h> 
#include <csignal>
#include <sys/prctl.h>  /* prctl */

// Local Function hoisting
unsigned long getTimestamp(struct Signal& s);
int mapSpeeds(unsigned char val, int pwm_min, int pwm_mid, int pwm_max);

// Signal handlers
static void sig_handler(int signum);
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

// Overridden with custom logic, alternative to the callbacks if only one TATS object
void control::TATS::onTargetUpdate(control::INFO info, EVENT eventType) {
    switch (eventType) {
        case EVENT::ON_UPDATE:
            if (info.tracking) {
                std::cout << "Currently tracked target: " << config->classes[info.id] << std::endl;
            }
            return;
        default:
            return;
    }
}

void control::TATS::onServoUpdate(double pan, double tilt) {
   std::cout << "Pan: " 
   << std::to_string(Utility::rescaleAction(pan, -45.0, 45.0)) 
   << ", Tilt: " 
   << std::to_string(Utility::rescaleAction(tilt, -45.0, 45.0)) 
   << std::endl;
}

int main() {
    signal(SIGTSTP, sig_handler);  // CTRL-Z on terminal (kill -SIGTSTP PID)

    parameters = new Utility::Parameter();
    config = new Utility::Config();

    // Setup eval mode options
    config->trainMode = false; /* Disable train mode */
    config->logOutput = false; /* Disable debug messages */

    wire = new control::Wire();
    pwm = new control::PCA9685(0x40, wire);
    servos = new control::ServoKit(pwm);
    targetTrackingSystem = new control::TATS(*config, servos);
        
    
    /*  
        Callbacks follow a different set of rules than onTargetUpdate() and onServoUpdate().
        1.) Its called asynchronously in another thread (will not slow detection or servo updates)
        2.) Called after servos AND the target updates. 
        3.) Allows for instance specific logic.
        
        As a result, callbacks have more computational overhead 
    
        Example: 
        std::function<void(control::INFO const&)> updateCallback = [](control::INFO event) {
            // TODO:: Logic that moves car towards the target given angle
        };
        targetTrackingSystem->registerCallback(control::EVENT::ON_UPDATE, updateCallback); 
    */

    targetTrackingSystem->init(1);     

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
                oldData = newData;
                radio.read(&newData.buffer, 8);

                if (getTimestamp(newData) != getTimestamp(oldData)) {
                    double wls = mapSpeeds(newData.buffer[data::wLeft], 750, 1500, 2250);
                    double wrs = mapSpeeds(newData.buffer[data::wRight], 750, 1500, 2250);
                    pwm->writeMicroseconds(10, wls);
                    pwm->writeMicroseconds(11, wrs);
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

int mapSpeeds(unsigned char val, int pwm_min=750, int pwm_mid=1500, int pwm_max=2250) {

    int adjusted;
    int max = 255;
    int middle = 127;
    int clipped = std::clamp(static_cast<int>(val), 0, max);

    // Not one-to-one mapping of microseconds on lower and upper speed ranges
    if (clipped <= middle) {
    adjusted = Utility::mapOutput(clipped, 0, middle, pwm_min, pwm_mid); 
    } else {
    adjusted = Utility::mapOutput(clipped, middle + 1, max, pwm_mid, pwm_max);
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