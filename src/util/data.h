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
    #define NUM_SERVOS 2
    #define NUM_INPUT 8
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
        double spf; // seconds per frame

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
        double obj;
        double frame;
        double e; 
        double spf;

        void getStateArray(double state[NUM_INPUT]) {

            double errorBound = pidStateData.setPoint * 2.0;
            double currentError = (obj - pidStateData.setPoint);
            double deltaTime = pidStateData.dt * 1000.0; // Back to mili

            /* 
                The State

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

                8.) Second per frame: T 
                    Note: In seconds
            */

            // Scale roughly between -1.0 ~ 1.0
            state[0] = currentError / errorBound;
            state[1] = ( pidStateData.i + ( currentError * pidStateData.dt )) / ( 2.0 * errorBound );
            state[2] = deltaTime > 0.0 ? (currentError - pidStateData.errors[0]) / deltaTime : 0.0;
            state[3] = deltaTime > 0.0 ? (currentError - ( 2.0 * pidStateData.errors[0] ) + pidStateData.errors[1]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
            state[4] = deltaTime > 0.0 ? (currentAngle - pidStateData.outputs[0]) / deltaTime : 0.0;
            state[5] = deltaTime > 0.0 ? (currentAngle - ( 2.0 * pidStateData.outputs[0] ) + pidStateData.outputs[1]) / std::pow<double>(deltaTime, 2.0) : 0.0; 
            state[6] = deltaTime > 0.0 ? pidStateData.dt : 0.0;
            state[7] = spf;
        } 

        PIDAndServoStateData() : 
            pidStateData({}),
            currentAngle(0.0), 
            lastAngle(0.0), 
            obj(0.0), 
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
        bool done;
        bool empty;

        TrainData() :
            currentState({}),
            nextState({}),
            reward(0.0),
            actions({0.0}),
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
        std::vector<std::string> targets;

        // PID options
        double pidOutputHigh;
        double pidOutputLow;
        double defaultGains[3];

        // Servo options
        struct control::Servo servoConfigurations[NUM_SERVOS];
        bool invertServo[NUM_SERVOS];
        bool disableServo[NUM_SERVOS];
        double resetAngles[NUM_SERVOS];
        double anglesHigh[NUM_SERVOS];
        double anglesLow[NUM_SERVOS];

        // Servo alternation in training reduces noise significantly. Faster convergence if enabled generally
        bool alternateServos;
        int alternateSteps;
        int alternateStop;
        int alternateEpisodeEndCap;

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
            episodeEndCap(true),                 // End episode early
            maxStepsPerEpisode(500),             // Max number of steps in an episode

            batchSize(256),                      // Network batch size.
            initialRandomActions(true),          // Enable random actions.
            numInitialRandomActions(2500),       // Number of random actions taken.
            trainMode(true),                     // When autotuning is on, 'false' means network test mode.
            useAutoTuning(true),                 // Use SAC network to query for PID gains.
            variableFPS(true),                   // Vary the FPS in training
            FPSVariance(5.0),                    // Average change in FPS
            varyFPSChance(0.5),                  // Percentage of frames that have variable FPS

            recheckFrequency(10),                // Num frames in-between revalidations of
            lossCountMax(1),                     // Max number of rechecks before episode is considered over
            updateRate(7),                       // Servo updates, update commands per second
            trainRate(1.0),					     // Network updates, sessions per second
            logOutput(true),                    // Prints various info to console
            
            disableServo({ true, false }),       // Disable the { Y, X } servos
            invertServo({ true, false }),        // Flip output angles { Y, X } servos
            resetAngles({ 0.0, 0.0 }),           // Angle when reset
            anglesHigh({ 40.0, 40.0 }),          // Max allowable output angle to servos
            anglesLow({ -40.0, -40.0 }),         // Min allowable output angle to servos

            servoConfigurations({                // Hardware settings for individual servos
                { 0, -56.5, 56.5, 0.750, 2.250, 0.0 }, 
                { 1, -56.5, 56.5, 0.750, 2.250, 0.0 } 
            }),
            
            alternateServos(false),              // Whether to alternate servos at the start of training
            alternateSteps(100),                 // Steps per servo (will increase exponentially as training proggresses (doubles threshold each time its met)). Cut short by 'alternateEpisodeEndCap'
            alternateStop(300),                  // Number of alternations
            alternateEpisodeEndCap(15),          // Number "end of episodes" before switching again. Prevents too much noise when both servos are enabled.

            trackerType(1),						 // { CSRT, MOSSE, GOTURN }
            useTracking(false),					 // Use openCV tracker instead of face detection
            draw(false),						 // Draw target bounding box and center on frame
            showVideo(false),					 // Show camera feed
            cascadeDetector(true),				 // Use faster cascade face detector
            usePIDs(true),                       // Network outputs PID gains, or network outputs angle directly
            actionHigh(0.2),                     // Max output to of policy network's logits
            actionLow(0.0),                      // Min output to of policy network's logits
            pidOutputHigh(40.0),                 // Max output allowed for PID's
            pidOutputLow(-40.0),				 // Min output allowed for PID's
            defaultGains({ 1.0, 1.0, 1.0 }),     // Gains fed to pids when initialized
            // targets({"0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"}),
            targets({"face"}),
            
            dims({ 720, 720 }),                  // Dimensions of frame
            maxFrameRate(60),                    // Camera capture rate
            multiProcess(true)                   // Enables autotuning in a seperate process. Otherwise its a thread.
            {}
    } typedef cfg;

    struct Parameter {

        int pid;
        bool isTraining;
        bool freshData;

        Parameter() {}
    } typedef param;
}