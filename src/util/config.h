#pragma once
/*
    This file should contain data formats, structs, unions, and typedefs.
    Keeps main.cpp cleaner.
*/
#include <vector>
#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/string.hpp>
#include "../pid/PID.h"
#include "../servo/ServoKit.h"

namespace Utility {
    #define NUM_SERVOS 2                         // Number of servos used 
    #define NUM_INPUT 22                         // Size of the state schema
    #define HISTORY_BUFFER_SIZE 6                // Number of errors/outputs to hold onto accross application
    #define NUM_HIDDEN 256                       // Number of nodes in each networks hidden layer
    #define USE_PIDS false                       // When enabled, AI directly computes angles angles are non-negative, from 0 to 180, otherwise -90 to 90.
    #define USE_POT false                        // Use predictive object location
    #define NUM_ACTIONS ((USE_PIDS) ? 3 : ((USE_POT) ? 2 : 1) ) 

    enum DetectorType {
        CASCADE, 
        ARUCO, 
        YOLO,
        RCNN
    };

    enum ActionType {
        PID,
        ANGLE,
        SPEED
    };

    struct EventData { 
        bool done;                               // For when the targer is lost
        double obj;                              // The X or Y center of object on in frame
        double size;                             // The bounding size of object along its X or Y axis
        double point;                            // The X or Y Origin coordinate of object
        double frame;                            // The X or Y center of frame
        double error;                            // Number of pixels between object and target centers
        double timestamp;                        // Unique identifier of this event
        double spf;                              // seconds per frame of the detect thread
        double tracking;                         // Flag that indicates if target has temporally vanished.

        void reset() {
            done = false;
            error = 0.0;
            obj = 0.0;
            size = 0.0;
            frame = 0.0;
            timestamp = 0.0;
            spf = 0.0;
            tracking = 0.0;
        }
        
        EventData() : 
            done(false), 
            error(0.0), 
            frame(0.0), 
            size(0.0),
            point(0.0),
            obj(0.0), 
            timestamp(0.0),
            spf(0.0),
            tracking(1.0)
        {}

        EventData( bool done, double error, double frame, double size, double point, double obj, double timestamp, double spf, double tracking)
        {
            this->done = done;
            this->error = error;
            this->frame = frame;
            this->size = size;
            this->point = point;
            this->obj = obj;
            this->timestamp = timestamp;
            this->spf = spf;
            this->tracking = tracking;
        }
    } typedef ED;

    // Current state of servo/pid
    struct StateData {
        struct PIDState pidStateData;
        double errors[HISTORY_BUFFER_SIZE] = { 0.0 };
        double outputs[HISTORY_BUFFER_SIZE] = { 0.0 };
        double deltaAngles[HISTORY_BUFFER_SIZE] = { 0.0 };
        double angleTimestamps[HISTORY_BUFFER_SIZE] = { 0.0 };
        double errorTimestamps[HISTORY_BUFFER_SIZE] = { 0.0 };
        ActionType actionType = ActionType::ANGLE;
        double currentAngle;
        double spf;  
        double tracking;

        void setData(
            double errs[HISTORY_BUFFER_SIZE], 
            double outs[HISTORY_BUFFER_SIZE], 
            double dltAng[HISTORY_BUFFER_SIZE], 
            double angleT[HISTORY_BUFFER_SIZE],
            double errorT[HISTORY_BUFFER_SIZE]
        ) {
            
            for (int i = 0; i < HISTORY_BUFFER_SIZE; i++) {
                errors[i] = errs[i];
                outputs[i] = outs[i];
                deltaAngles[i] = dltAng[i];
                angleTimestamps[i] = angleT[i];
                errorTimestamps[i] = errorT[i];
            }
        }

        void getStateArray(double state[NUM_INPUT]) {
            double errorBound = pidStateData.setPoint * 2.0;
            double deltaTime = pidStateData.dt * 1000.0; // Back to mili
            double currentError = (errors[0] - pidStateData.setPoint);
            
            switch (actionType)
            {
            case ActionType::PID:
                state[0] = outputs[0];
                state[1] = errors[0];
                state[2] = outputs[1];
                state[3] = errors[1];
                state[4] = outputs[2];
                state[5] = errors[2];
                state[6] = outputs[3];
                state[7] = errors[3];
                state[8] = outputs[3];
                state[9] = errors[3];
                state[10] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                state[11] = deltaTime > 0.0 ? spf : 0.0;
                state[12] = tracking;
                break;

                /* 
            
                    state[0] = currentAngle;
                    state[1] = currentError / errorBound;
                    state[2] = ( pidStateData.i + ( currentError * pidStateData.dt )) / ( 2.0 * errorBound );
                    state[3] = deltaTime > 0.0 ? (currentError - pidStateData.errors[0]) / deltaTime : 0.0;
                    state[4] = deltaTime > 0.0 ? (currentError - ( 2.0 * pidStateData.errors[0] ) + pidStateData.errors[1]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
                    state[5] = deltaTime > 0.0 ? (pidStateData.outputs[0] - pidStateData.outputs[1]) / deltaTime : 0.0;
                    state[6] = deltaTime > 0.0 ? (pidStateData.outputs[0] - ( 2.0 * pidStateData.outputs[1] ) + pidStateData.outputs[2]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
                    state[7] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                    state[8] = spf;

                    1.) Current Error

                    2.) Integral error: 
                        Formula: Sum (t_i * T)
                        Note: For last 5 errors

                    3.) First order delta error: 
                        Formula: (E(t_i) - E(t_i + 1)) / T

                    4.) Second order delta error: 
                        Formula: (E(t_i) - ( 2.0 * E(t_i + 1) ) + E(t_i + 2)) / T^2 

                    5.) First order delta angle: 
                        Formula: (A(t_i) - A(t_i + 1)) / T

                    6.) Second order delta angle: 
                        Formula: (A(t_i) - ( 2.0 * A(t_i + 1) ) + A(t_i + 2)) / T^2 

                    7.) Delta time: T 
                        Note: In seconds

                    8.) Seconds per frame: T 
        
                */
                
                break;
            
            case ActionType::ANGLE:
                state[0] = outputs[0];
                state[1] = errors[0];
                state[2] = outputs[1];
                state[3] = errors[1];
                state[4] = outputs[2];
                state[5] = errors[2];
                state[6] = outputs[3];
                state[7] = errors[3];
                state[8] = outputs[3];
                state[9] = errors[3];
                state[10] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                state[11] = deltaTime > 0.0 ? spf : 0.0;
                state[12] = tracking;
                break;
            case ActionType::SPEED:
                state[0] = errors[0];
                state[1] = deltaAngles[0];
                state[2] = angleTimestamps[0];
                state[3] = errorTimestamps[0];
                state[4] = errors[1];
                state[5] = deltaAngles[1];
                state[6] = angleTimestamps[1];
                state[7] = errorTimestamps[1];
                state[8] = errors[2];
                state[9] = deltaAngles[2];
                state[10] = angleTimestamps[2];
                state[11] = errorTimestamps[2];
                state[12] = errors[3];
                state[13] = deltaAngles[3];
                state[14] = angleTimestamps[3];
                state[15] = errorTimestamps[3];
                state[16] = errors[4];
                state[17] = deltaAngles[4];
                state[18] = angleTimestamps[4];
                state[19] = errorTimestamps[4];
                // state[20] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                state[20] = deltaTime > 0.0 ? spf : 0.0;
                state[21] = tracking;
                break;
            default:
                throw std::runtime_error("Invalid action type provided");
            }
        } 

        StateData() : 
            pidStateData({}),
            errors({ 0.0 }),
            outputs({ 0.0 }),
            deltaAngles({ 0.0 }),
            actionType(Utility::ActionType::ANGLE),
            currentAngle(0.0),
            spf(0.0),
            tracking(1.0)
        {}

    } typedef SD;

    // Result Data for taking a step with for a given servo/PID
    struct TrainData {
        ActionType actionType;
        SD currentState;
        SD nextState;
        double reward;
        double actions[NUM_ACTIONS];
        double errors[2];
        bool done;
        bool empty;

        void setActionType(ActionType at) {
            actionType = at;
            currentState.actionType = actionType;
            nextState.actionType = actionType;
        }

        TrainData() :
            currentState({}),
            nextState({}),
            reward(0.0),
            actions({0.0}),
            errors({0.0}),
            done(false),
            empty(true)
        {}
    } typedef TD;

    // Results of taking actions for each PID/servos in env
    struct StepResults {
        void setActionType(ActionType at) {
            for (TD &servo : servos) {
                servo.setActionType(at);
            }
        }
        TD servos[NUM_SERVOS];
        StepResults() :
            servos({})
        {}
    } typedef SR;

    // State of PIDs/Servos after reset
    struct ResetData {
        SD servos[NUM_SERVOS];

        void setActionType(ActionType at) {
            for (SD &servo : servos) {
                servo.actionType = at;
            }
        }

        ResetData() :
            servos({})
        {}
    } typedef RD;

    typedef boost::interprocess::allocator<TD, boost::interprocess::managed_shared_memory::segment_manager> ShmemAllocator;
    typedef boost::interprocess::allocator<char, boost::interprocess::managed_shared_memory::segment_manager> CharAllocator;
    typedef boost::interprocess::basic_string<char, std::char_traits<char>, CharAllocator> sharedString;
    typedef boost::interprocess::vector<TD, ShmemAllocator> SharedBuffer;
    typedef std::vector<TD> TrainBuffer;

    struct Config {

        // Network Options
        int numActions;
        int numHidden;
        int numInput;
        double actionHigh;
        double actionLow;
        ActionType actionType;

        // Yolo detection options
        DetectorType detector;
        std::string detectorPath;
        std::vector<std::string> targets;
        std::vector<std::string> classes;

        // Train Options
        long maxTrainingSteps;
        int maxBufferSize;
        int minBufferSize;
        int updateRate;
        int maxTrainingSessions;
        int batchSize;
        int stepsWithPretrainedModel;
        int numInitialRandomActions;
        int numTransferLearningSteps;
        bool trainMode;
        bool initialRandomActions;
        bool episodeEndCap;
        int maxStepsPerEpisode;
        double numUpdates;
        double trainRate;
        bool logOutput;
        bool variableFPS;
        int FPSVariance;
        double varyFPSChance;
        double resetAngleVariance;
        double resetAngleChance;
        bool varyResetAngles;
        bool useCurrentAngleForReset;
        
        // Tracking Options
        int lossCountMax;
        int recheckFrequency;
        int trackerType;
        bool useTracking;
        bool useAutoTuning;
        bool draw;
        bool showVideo;
        bool usePIDs;
        bool usePOT;
        int resetAfterNInnactiveFrames;
        int maxPredictiveSteps;

        // PID options
        double pidOutputHigh;
        double pidOutputLow;
        double defaultGains[3];

        // Servo options
        struct control::Servo servoConfigurations[NUM_SERVOS];
        bool invertData[NUM_SERVOS];
        bool invertAngles[NUM_SERVOS];
        bool disableServo[NUM_SERVOS];
        double resetAngles[NUM_SERVOS];
        double anglesHigh[NUM_SERVOS];
        double anglesLow[NUM_SERVOS];
        double anglesRange[NUM_SERVOS];
        double minServoSpeed;
        double maxDeltaAngle;

        // Other 
        int dims[2];
        int captureSize[2];
        int fps;
        bool multiProcess;
    
        Config() :

            numActions(NUM_ACTIONS),             // Number of output classifications for policy network
            numHidden(NUM_HIDDEN),               // Number of nodes in the hidden layers of each network
            numInput(NUM_INPUT),                 // Number of elements in policy/value network's input vectors
            actionType(ActionType::ANGLE),       // How output actions of network are used. 

            maxTrainingSessions(1),              // Number of training sessions on model params
            maxBufferSize(500000),               // Max size of buffer. When full, oldest elements are kicked out.
            minBufferSize(2000),                 // Min replay buffer size before training size.
            maxTrainingSteps(500000),			 // Max training steps agent takes.
            numUpdates(5),                       // Num updates per training session.
            episodeEndCap(true),                 // End episode early
            maxStepsPerEpisode(250),             // Max number of steps in an episode

            batchSize(512),                      // Network batch size.
            initialRandomActions(true),          // Enable random actions.
            numInitialRandomActions(2500),       // Number of random actions taken.
            stepsWithPretrainedModel(false),     // After random steps, uses loaded save to perform steps in evaluation mode
            numTransferLearningSteps(5000),      // Number of steps to take on a pre-trained model in evaluation mode, done after random steps
            trainMode(false),                    // When autotuning is on, 'false' means network test mode.
            useAutoTuning(true),                 // Use SAC network to query for PID gains.
            variableFPS(true),                   // Vary the FPS in training
            FPSVariance(45),                     // Average change in FPS
            resetAngleVariance(40.0),            // In training, the degree of variance in reset angles
            resetAngleChance(0.05),              // Chance to randomly chance the current angle the servos are wating at
            varyResetAngles(true),               // vary reset angles diring training
            recheckFrequency(20),                // Num frames in-between revalidations of
            lossCountMax(2),                     // Max number of rechecks before episode is considered over. 
                                                 // In the case of usePOT, MAX uses of predictive object tracking.
            updateRate(8),                       // Servo updates, update commands per second
            trainRate(.25),					     // Network updates, sessions per second
            logOutput(true),                     // Prints various info to console
            
            disableServo({ false, false }),       // Disable the { Y, X } servos
            invertData({ true, false }),         // Flip input data { Y, X } servos
            invertAngles({ false, false }),      // Flip output angles { Y, X } servos
            resetAngles({                        // Angle when reset
                0.0, 0.0
            }),    
            anglesHigh({                         // Max allowable output angle to servos
                72.0, 72.0
            }),          
            anglesLow({                          // Min allowable output angle to servos
                -72.0, -72.0
            }), 
            anglesRange({                        // Total range of the servos
                90.0, 90.0
            }),   
            servoConfigurations(                 // Hardware settings for individual servos         
                {                             
                    { 0, -72.0, 72.0, 0.750, 2.250, 0.0, .14}, 
                    { 1, -72.0, 72.0, 0.750, 2.250, 0.0, .14 } // D951TW
                }

                // {                             
                //     { 0, -56.5, 56.5, 0.750, 2.250, 0.0 }, 
                //     { 1, -56.5, 56.5, 0.750, 2.250, 0.0 } // HS7985MG
                // }

                // {                             
                //     { 0, -65.0, 65.0, 0.900, 2.1, 0.0 }, 
                //     { 1, -65.0, 65.0, 0.900, 2.1, 0.0 } // SB2272MG
                // }
            ),    
            minServoSpeed(.20),                  // 60 deg / sec. Choose servo rated below this value (faster servo)
            maxDeltaAngle(15.0),                 // Max angular change made by servos in ActionType::SPEED mode
            trackerType(1),						 // { CSRT, MOSSE, GOTURN }
            useTracking(false),					 // Use openCV tracker instead of face detection
            usePOT(USE_POT),                     // Predictive Object Tracking. If detection has failed, uses AI to predict objects next location
            resetAfterNInnactiveFrames(25),      // Reset to default angles after N frames. -1 indicates never resetting. 
            maxPredictiveSteps(                  // When target disappears, Number of times AI can guess its new action
                5
            ),
            useCurrentAngleForReset(true),       // Use current angle as reset angle when target has lost track
            draw(false),  					     // Draw target bounding box and center on frame
            showVideo(false),					 // Show camera feed
            usePIDs(USE_PIDS),                   // Network outputs PID gains, or network outputs angle directly
            actionHigh(                          // Max output to of policy network's logits   
                USE_PIDS ? 0.1 : 45.0
            ),                     
            actionLow(                           // Min output to of policy network's logits
                USE_PIDS ? 0.0 : -45.0
            ),                      
            pidOutputHigh(45.0),                 // Max output allowed for PID's
            pidOutputLow(-45.0),				 // Min output allowed for PID's
            defaultGains({ 0.04, 0.02, 0.001 }), // Gains fed to pids when initialized
            
            dims({ 1080, 1080 }),                // The image crop dimensions. Applied before autotuning input.
            captureSize({ 1080, 1920 }),         // The dimensions for capture device
            
            fps(60),                             // Camera capture rate
            multiProcess(true),                  // Enables autotuning in a seperate process. Otherwise its a thread.
            
            // detector(DetectorType::ARUCO),
            // detectorPath(""),
            // targets({"ArUco"}),
            // classes({"ArUco"})           
            
            // detector(DetectorType::CASCADE),
            // detectorPath("/models/haar/haarcascade_frontalface_default.xml"),
            // targets({"face"}),
            // classes({"face"})

            detector(DetectorType::YOLO),
            detectorPath("/models/yolo/yolov5s6.uno.640.torchscript.pt"),
            targets({"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"}),
            classes({"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"})

            // detector(DetectorType::YOLO),
            // detectorPath("/models/yolo/yolov5s6.coco.640.torchscript.pt"),
            // targets({ "cat", "dog", "bottle", "cup"}),
            // classes({ 
            // "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            // "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            // "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            // "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            // "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            // "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            // "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            // "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            // "hair drier", "toothbrush" })
            {}
    } typedef cfg;

    struct Parameter {

        int pid;
        bool isTraining;
        bool freshData;

        Parameter() {}
    } typedef param;
}