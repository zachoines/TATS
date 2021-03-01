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
    #define NUM_INPUT 10                         // Size of the state schema
    #define ERROR_LIST_SIZE 5                    // Number of errors/outputs to hold onto accross application
    #define NUM_HIDDEN 256                       // Number of nodes in each networks hidden layer
    #define USE_PIDS 0                           // When enabled, AI directly computes angles angles are non-negative, from 0 to 180, otherwise -90 to 90.
    #define USE_POT 1                            // Use predictive object location
    #define NUM_ACTIONS ((USE_PIDS) ? 3 : ((USE_POT) ? 2 : 1) ) 

    struct EventData { 

        bool done;                               // For when the targer is lost
        double obj;                              // The X or Y center of object on in frame
        double size;                             // The bounding size of object along its X or Y axis
        double point;                            // The X or Y Origin coordinate of object
        double frame;                            // The X or Y center of frame
        double error;                            // Number of pixels between object and target centers
        double timestamp;                      
        double spf;                              // seconds per frame of the detect thread

        void reset() {
            done = false;
            error = 0.0;
            obj = 0.0;
            size = 0.0;
            frame = 0.0;
            timestamp = 0.0;
            spf = 0.0;
        }
        
        EventData() : 
            done(false), 
            error(0.0), 
            frame(0.0), 
            size(0.0),
            point(0.0),
            obj(0.0), 
            timestamp(0.0),
            spf(0.0)
        {}
    } typedef ED;

    // Current state of servo/pid
    struct PIDAndServoStateData {
        struct PIDState pidStateData;
        double currentAngle;
        double lastAngle;
        double angleMax;
        double errors[ERROR_LIST_SIZE] = { 0.0 };
        double outputs[ERROR_LIST_SIZE] = { 0.0 };
        double obj;
        double frame;
        double e; 
        double spf;

        void setData(double errs[ERROR_LIST_SIZE], double outs[ERROR_LIST_SIZE]) {
            for (int i = 0; i < 5; i++) {
                errors[i] = errs[i];
                outputs[i] = outs[i];
            }
        }

        void getStateArray(double state[NUM_INPUT]) {

            double errorBound = pidStateData.setPoint * 2.0;
            double currentError = (obj - pidStateData.setPoint);
            double deltaTime = pidStateData.dt * 1000.0; // Back to mili

            /* 
                The State (then multiplied by constants to move closer to -1 ~ 1)

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

            if (USE_PIDS) {
                state[0] = currentAngle;
                state[1] = currentError / errorBound;
                state[2] = ( pidStateData.i + ( currentError * pidStateData.dt )) / ( 2.0 * errorBound );
                state[3] = deltaTime > 0.0 ? (currentError - pidStateData.errors[0]) / deltaTime : 0.0;
                state[4] = deltaTime > 0.0 ? (currentError - ( 2.0 * pidStateData.errors[0] ) + pidStateData.errors[1]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
                state[5] = deltaTime > 0.0 ? (pidStateData.outputs[0] - pidStateData.outputs[1]) / deltaTime : 0.0;
                state[6] = deltaTime > 0.0 ? (pidStateData.outputs[0] - ( 2.0 * pidStateData.outputs[1] ) + pidStateData.outputs[2]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
                state[7] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                state[8] = spf;
            } else {
                // double integral = 0.0;
                // for (int i = 0; i < 5; i++) {
                //     integral += (errors[i] * pidStateData.dt);
                // }

                // state[0] = outputs[0];
                // state[1] = errors[0];
                // state[2] = integral;
                // state[3] = deltaTime > 0.0 ? ((errors[0] - errors[1]) / deltaTime) * 100.0 : 0.0;
                // state[4] = deltaTime > 0.0 ? ((errors[0] - ( 2.0 * errors[1] ) + errors[2]) / std::pow<double>(deltaTime, 2.0)) * 100.0 : 0.0; 
                // state[5] = deltaTime > 0.0 ? ((outputs[0] - outputs[1]) / deltaTime) * 100.0 : 0.0;
                // state[6] = deltaTime > 0.0 ? ((outputs[0] - ( 2.0 * outputs[1] ) + outputs[2]) / std::pow<double>(deltaTime, 2.0)) * 100.0 : 0.0; 
                // state[7] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                // state[8] = deltaTime > 0.0 ? spf : 0.0;

                state[0] = outputs[0];
                state[1] = errors[0];
                state[2] = outputs[1];
                state[3] = errors[1];
                state[4] = outputs[2];
                state[5] = errors[2];
                state[6] = outputs[3];
                state[7] = errors[3];
                state[8] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
                state[9] = deltaTime > 0.0 ? spf : 0.0;
            }
        } 

        PIDAndServoStateData() : 
            pidStateData({}),
            currentAngle(0.0), 
            lastAngle(0.0), 
            obj(0.0), 
            errors({ 0.0 }),
            outputs({ 0.0 }),
            frame(0.0), 
            e(0.0),
            spf(0.0)
        {}

    } typedef SD;

    // State of PIDs/Servos after reset
    struct ResetData {
        SD servos[NUM_SERVOS];
        ResetData() : 
            servos({{}})
        {}
    } typedef RD;

    // Result Data for taking a step with for a given servo/PID
    struct TrainData {
        SD currentState;
        SD nextState;
        double reward;
        double actions[NUM_ACTIONS];
        double errors[2];
        bool done;
        bool empty;

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
        TD servos[NUM_SERVOS];
        StepResults() :
            servos({{}})
        {}
    } typedef SR;

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

        // Yolo detection options
        std::string yoloPath;
        std::vector<std::string> targets;
        std::vector<std::string> classes;

        // Train Options
        long maxTrainingSteps;
        int maxBufferSize;
        int minBufferSize;
        int updateRate;
        int maxTrainingSessions;
        int batchSize;
        int numInitialRandomActions;
        bool trainMode;
        bool initialRandomActions;
        bool episodeEndCap;
        int maxStepsPerEpisode;
        double numUpdates;
        double trainRate;
        bool logOutput;
        bool variableFPS;
        double FPSVariance;
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
        bool cascadeDetector;
        bool usePIDs;
        bool usePOT;
        int resetAfterNInnactiveFrames;

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

        // Other 
        int dims[2];
        int captureSize[2];
        int resize[2];
        int fps;
        bool multiProcess;
    
        Config() :

            numActions(NUM_ACTIONS),             // Number of output classifications for policy network
            numHidden(NUM_HIDDEN),               // Number of nodes in the hidden layers of each network
            numInput(NUM_INPUT),                 // Number of elements in policy/value network's input vectors

            maxTrainingSessions(1),              // Number of training sessions on model params
            maxBufferSize(500000),               // Max size of buffer. When full, oldest elements are kicked out.
            minBufferSize(2000),                 // Min replay buffer size before training size.
            maxTrainingSteps(500000),			 // Max training steps agent takes.
            numUpdates(5),                       // Num updates per training session.
            episodeEndCap(true),                 // End episode early
            maxStepsPerEpisode(500),             // Max number of steps in an episode

            batchSize(128),                      // Network batch size.
            initialRandomActions(true),          // Enable random actions.
            numInitialRandomActions(2500),       // Number of random actions taken.
            trainMode(false),                    // When autotuning is on, 'false' means network test mode.
            useAutoTuning(true),                 // Use SAC network to query for PID gains.
            variableFPS(true),                   // Vary the FPS in training
            FPSVariance(7.0),                    // Average change in FPS
            varyFPSChance(0.5),                  // Percentage of frames that have variable FPS
            resetAngleVariance(30.0),            // In training, the degree of variance in reset angles
            resetAngleChance(0.05),              // Chance to randomly chance the current angle the servos are wating at
            varyResetAngles(true),               // vary reset angles diring training

            recheckFrequency(120),               // Num frames in-between revalidations of
            lossCountMax(30),                    // Max number of rechecks before episode is considered over. 
                                                 // In the case of usePOT, MAX uses of predictive object tracking.
            updateRate(4),                       // Servo updates, update commands per second
            trainRate(.25),					     // Network updates, sessions per second
            logOutput(true),                     // Prints various info to console
            
            disableServo({ false, false  }),       // Disable the { Y, X } servos
            invertData({ false, true }),         // Flip input data { Y, X } servos
            invertAngles({ false, false }),      // Flip output angles { Y, X } servos
            resetAngles({                        // Angle when reset
                0.0, 0.0
            }),    
            anglesHigh({                         // Max allowable output angle to servos
                45.0, 45.0
            }),          
            anglesLow({                          // Min allowable output angle to servos
                -45.0, -45.0
            }),         
            servoConfigurations(                 // Hardware settings for individual servos         
                {                             
                    { 0, -56.5, 56.5, 0.750, 2.250, 0.0 }, 
                    { 1, -56.5, 56.5, 0.750, 2.250, 0.0 } 
                }
            ),    

            trackerType(1),						 // { CSRT, MOSSE, GOTURN }
            useTracking(false),					 // Use openCV tracker instead of face detection
            usePOT((bool)USE_POT),               // Predictive Object Tracking. If detection has failed, uses AI to predict objects next location
            // usePOT(false),
            resetAfterNInnactiveFrames(10),      // Reset to default angles after N frames. -1 indicates never resetting. 
            useCurrentAngleForReset(true),       // Use current angle as reset angle when target has lost track
            draw(false),						 // Draw target bounding box and center on frame
            showVideo(false),					 // Show camera feed
            cascadeDetector(false),				 // Use faster cascade face detector
            usePIDs((bool)USE_PIDS),             // Network outputs PID gains, or network outputs angle directly
            actionHigh(                          // Max output to of policy network's logits   
                USE_PIDS ? 0.1 : 45.0
            ),                     
            actionLow(                           // Min output to of policy network's logits
                USE_PIDS ? 0.0 : -45.0
            ),                      
            pidOutputHigh(45.0),                 // Max output allowed for PID's
            pidOutputLow(-45.0),				 // Min output allowed for PID's
            defaultGains({ 0.08, 0.04, 0.002}),  // Gains fed to pids when initialized
            
            dims({ 720, 720 }),                  // The image crop dimensions. Applied before autotuning input.
            captureSize({ 720, 1280 }),          // The dimensions for capture device
            resize({ 720, 1280 }),               // The dimensions to scale to before cropping
            
            fps(60),                             // Camera capture rate
            multiProcess(true),                  // Enables autotuning in a seperate process. Otherwise its a thread.
            // targets({"face"}),
            targets({"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"}),
            classes({"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"}),            
            yoloPath("/models/yolo/yolo5s_uno.torchscript.pt")
            // yoloPath("/models/yolo/yolov5s_coco.torchscript.pt"),
            // targets({ "person" }),
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