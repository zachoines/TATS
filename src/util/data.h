  
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

namespace Utility {
    #define NUM_SERVOS 2
    #define NUM_INPUT 6
    #define NUM_ACTIONS 3
    #define NUM_HIDDEN 256
    
    struct EventData { 

        bool done;
        double obj; // The X or Y center of object on in frame
        double size; // The bounding size of object along its X or Y axis
        double point; // The X or Y Origin coordinate of object
        double frame; // The X or Y center of frame
        double error;
        double timestamp;

        void reset() {
            done = false;
            error = 0.0;
            obj = 0.0;
            size = 0.0;
            frame = 0.0;
            timestamp = 0.0;
        }
        EventData() : done(false), error(0.0), frame(0.0), obj(0.0), timestamp(0.0) {}
    } typedef ED;

    // Current state of servo/pid
    struct PIDAndServoStateData {
        struct PIDState pidStateData;
        double currentAngle;
        double lastAngle;
        double obj;

        void getStateArray(double state[NUM_INPUT]) {
            pidStateData.getStateArray(state); // first 5 elems will be filled
            state[5] = lastAngle - currentAngle;
        }

    } typedef SD;

    // State of PIDs/Servos after reset
    struct ResetData {
        SD servos[NUM_SERVOS];
    } typedef RD;

    // Result Data for taking a step with for a given servo/PID
    struct TrainData {
        SD currentState;
        SD nextState;
        double reward;
        double actions[NUM_ACTIONS];
        bool done;
    } typedef TD;

    // Results of taking actions for each PID/servos in env
    struct StepResults {
        TD servos[NUM_SERVOS];
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

        // Train Options
        long maxTrainingSteps;
        int maxBufferSize;
        int minBufferSize;
        int updateRate;
        int maxTrainingSessions;
        int numFrameSkip;
        int batchSize;
        int numInitialRandomActions;
        bool trainMode;
        bool frameSkip;
        bool initialRandomActions;
        double numUpdates;
        double trainRate;
        
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

        // PID options
        double pidOutputHigh;
        double pidOutputLow;
        double defaultGains[3];

        // Servo options
        double angleHigh;
        double angleLow;
        double resetAngleX;
        double resetAngleY;
        bool invertX;
        bool invertY;
        bool disableX;
        bool disableY;

        // Other 
        int dims[2];
        int maxFrameRate;
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

            batchSize(256),                      // Network batch size.
            initialRandomActions(true),          // Enable random actions.
            numInitialRandomActions(10000),      // Number of random actions taken.
            trainMode(true),                     // When autotuning is on, 'false' means network test mode.
            useAutoTuning(true),                 // Use SAC network to query for PID gains.

            recheckFrequency(15),                // Num frames in-between revalidations of tracking quality
            lossCountMax(2),                     // Max number of rechecks before episode is considered over
            updateRate(5),                       // Servo updates, commands per second
            trainRate(.5),					     // Network updates, sessions per second
            invertX(false),                      // Flip output angles for pan
            invertY(false),						 // Flip output angles for tilt
            disableX(false),                     // Disable the pan servo
            disableY(true),                      // Disable the tilt servo

            trackerType(1),						 // { CSRT, MOSSE, GOTURN } 
            useTracking(false),					 // Use openCV tracker instead of face detection
            draw(false),						 // Draw target bounding box and center on frame
            showVideo(false),					 // Show camera feed
            cascadeDetector(true),				 // Use faster cascade face detector 
            usePIDs(true),                       // Network outputs PID gains, or network outputs angle directly
            actionHigh(.10),                     // Max output to of policy network's logits
            actionLow(0.0),                      // Min output to of policy network's logits        
            pidOutputHigh(60.0),                 // Max output allowed for PID's
            pidOutputLow(-60.0),				 // Min output allowed for PID's
            defaultGains({ 0.05, 0.04, 0.001 }), // Gains fed to pids when initialized
            angleHigh(60.0),                     // Max allowable output angle to servos
            angleLow(-60.0),                     // Min allowable output angle to servos
            resetAngleX(0.0),                    // Angle when reset
            resetAngleY(15.0),                   // Angle when reset
            // dims({ 2464, 3280 }),
            dims({ 720, 1280 }),                 // Dimensions of frame
            maxFrameRate(120),                   // Camera capture rate
            multiProcess(false)                  // Enables autotuning in a seperate process. Otherwise its a thread.
            {}
    } typedef cfg;

    struct Parameter {

        int pid;
        bool isTraining;
        bool freshData;

        Parameter() {}
    } typedef param;

}

