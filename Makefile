# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zachoines/Documents/repos/TATS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zachoines/Documents/repos/TATS

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/zachoines/Documents/repos/TATS/CMakeFiles /home/zachoines/Documents/repos/TATS/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/zachoines/Documents/repos/TATS/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named TATS

# Build rule for target.
TATS: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 TATS
.PHONY : TATS

# fast build rule for target.
TATS/fast:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/build
.PHONY : TATS/fast

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/main.cpp.s
.PHONY : main.cpp.s

src/detection/CascadeDetector.o: src/detection/CascadeDetector.cpp.o

.PHONY : src/detection/CascadeDetector.o

# target to build an object file
src/detection/CascadeDetector.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/CascadeDetector.cpp.o
.PHONY : src/detection/CascadeDetector.cpp.o

src/detection/CascadeDetector.i: src/detection/CascadeDetector.cpp.i

.PHONY : src/detection/CascadeDetector.i

# target to preprocess a source file
src/detection/CascadeDetector.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/CascadeDetector.cpp.i
.PHONY : src/detection/CascadeDetector.cpp.i

src/detection/CascadeDetector.s: src/detection/CascadeDetector.cpp.s

.PHONY : src/detection/CascadeDetector.s

# target to generate assembly for a file
src/detection/CascadeDetector.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/CascadeDetector.cpp.s
.PHONY : src/detection/CascadeDetector.cpp.s

src/detection/ObjectDetector.o: src/detection/ObjectDetector.cpp.o

.PHONY : src/detection/ObjectDetector.o

# target to build an object file
src/detection/ObjectDetector.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/ObjectDetector.cpp.o
.PHONY : src/detection/ObjectDetector.cpp.o

src/detection/ObjectDetector.i: src/detection/ObjectDetector.cpp.i

.PHONY : src/detection/ObjectDetector.i

# target to preprocess a source file
src/detection/ObjectDetector.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/ObjectDetector.cpp.i
.PHONY : src/detection/ObjectDetector.cpp.i

src/detection/ObjectDetector.s: src/detection/ObjectDetector.cpp.s

.PHONY : src/detection/ObjectDetector.s

# target to generate assembly for a file
src/detection/ObjectDetector.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/ObjectDetector.cpp.s
.PHONY : src/detection/ObjectDetector.cpp.s

src/detection/RCNNDetector.o: src/detection/RCNNDetector.cpp.o

.PHONY : src/detection/RCNNDetector.o

# target to build an object file
src/detection/RCNNDetector.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/RCNNDetector.cpp.o
.PHONY : src/detection/RCNNDetector.cpp.o

src/detection/RCNNDetector.i: src/detection/RCNNDetector.cpp.i

.PHONY : src/detection/RCNNDetector.i

# target to preprocess a source file
src/detection/RCNNDetector.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/RCNNDetector.cpp.i
.PHONY : src/detection/RCNNDetector.cpp.i

src/detection/RCNNDetector.s: src/detection/RCNNDetector.cpp.s

.PHONY : src/detection/RCNNDetector.s

# target to generate assembly for a file
src/detection/RCNNDetector.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/RCNNDetector.cpp.s
.PHONY : src/detection/RCNNDetector.cpp.s

src/detection/YoloDetector.o: src/detection/YoloDetector.cpp.o

.PHONY : src/detection/YoloDetector.o

# target to build an object file
src/detection/YoloDetector.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/YoloDetector.cpp.o
.PHONY : src/detection/YoloDetector.cpp.o

src/detection/YoloDetector.i: src/detection/YoloDetector.cpp.i

.PHONY : src/detection/YoloDetector.i

# target to preprocess a source file
src/detection/YoloDetector.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/YoloDetector.cpp.i
.PHONY : src/detection/YoloDetector.cpp.i

src/detection/YoloDetector.s: src/detection/YoloDetector.cpp.s

.PHONY : src/detection/YoloDetector.s

# target to generate assembly for a file
src/detection/YoloDetector.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/detection/YoloDetector.cpp.s
.PHONY : src/detection/YoloDetector.cpp.s

src/env/Env.o: src/env/Env.cpp.o

.PHONY : src/env/Env.o

# target to build an object file
src/env/Env.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/env/Env.cpp.o
.PHONY : src/env/Env.cpp.o

src/env/Env.i: src/env/Env.cpp.i

.PHONY : src/env/Env.i

# target to preprocess a source file
src/env/Env.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/env/Env.cpp.i
.PHONY : src/env/Env.cpp.i

src/env/Env.s: src/env/Env.cpp.s

.PHONY : src/env/Env.s

# target to generate assembly for a file
src/env/Env.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/env/Env.cpp.s
.PHONY : src/env/Env.cpp.s

src/network/Normal.o: src/network/Normal.cpp.o

.PHONY : src/network/Normal.o

# target to build an object file
src/network/Normal.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/Normal.cpp.o
.PHONY : src/network/Normal.cpp.o

src/network/Normal.i: src/network/Normal.cpp.i

.PHONY : src/network/Normal.i

# target to preprocess a source file
src/network/Normal.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/Normal.cpp.i
.PHONY : src/network/Normal.cpp.i

src/network/Normal.s: src/network/Normal.cpp.s

.PHONY : src/network/Normal.s

# target to generate assembly for a file
src/network/Normal.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/Normal.cpp.s
.PHONY : src/network/Normal.cpp.s

src/network/PolicyNetwork.o: src/network/PolicyNetwork.cpp.o

.PHONY : src/network/PolicyNetwork.o

# target to build an object file
src/network/PolicyNetwork.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/PolicyNetwork.cpp.o
.PHONY : src/network/PolicyNetwork.cpp.o

src/network/PolicyNetwork.i: src/network/PolicyNetwork.cpp.i

.PHONY : src/network/PolicyNetwork.i

# target to preprocess a source file
src/network/PolicyNetwork.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/PolicyNetwork.cpp.i
.PHONY : src/network/PolicyNetwork.cpp.i

src/network/PolicyNetwork.s: src/network/PolicyNetwork.cpp.s

.PHONY : src/network/PolicyNetwork.s

# target to generate assembly for a file
src/network/PolicyNetwork.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/PolicyNetwork.cpp.s
.PHONY : src/network/PolicyNetwork.cpp.s

src/network/QNetwork.o: src/network/QNetwork.cpp.o

.PHONY : src/network/QNetwork.o

# target to build an object file
src/network/QNetwork.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/QNetwork.cpp.o
.PHONY : src/network/QNetwork.cpp.o

src/network/QNetwork.i: src/network/QNetwork.cpp.i

.PHONY : src/network/QNetwork.i

# target to preprocess a source file
src/network/QNetwork.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/QNetwork.cpp.i
.PHONY : src/network/QNetwork.cpp.i

src/network/QNetwork.s: src/network/QNetwork.cpp.s

.PHONY : src/network/QNetwork.s

# target to generate assembly for a file
src/network/QNetwork.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/QNetwork.cpp.s
.PHONY : src/network/QNetwork.cpp.s

src/network/ReplayBuffer.o: src/network/ReplayBuffer.cpp.o

.PHONY : src/network/ReplayBuffer.o

# target to build an object file
src/network/ReplayBuffer.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ReplayBuffer.cpp.o
.PHONY : src/network/ReplayBuffer.cpp.o

src/network/ReplayBuffer.i: src/network/ReplayBuffer.cpp.i

.PHONY : src/network/ReplayBuffer.i

# target to preprocess a source file
src/network/ReplayBuffer.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ReplayBuffer.cpp.i
.PHONY : src/network/ReplayBuffer.cpp.i

src/network/ReplayBuffer.s: src/network/ReplayBuffer.cpp.s

.PHONY : src/network/ReplayBuffer.s

# target to generate assembly for a file
src/network/ReplayBuffer.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ReplayBuffer.cpp.s
.PHONY : src/network/ReplayBuffer.cpp.s

src/network/SACAgent.o: src/network/SACAgent.cpp.o

.PHONY : src/network/SACAgent.o

# target to build an object file
src/network/SACAgent.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/SACAgent.cpp.o
.PHONY : src/network/SACAgent.cpp.o

src/network/SACAgent.i: src/network/SACAgent.cpp.i

.PHONY : src/network/SACAgent.i

# target to preprocess a source file
src/network/SACAgent.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/SACAgent.cpp.i
.PHONY : src/network/SACAgent.cpp.i

src/network/SACAgent.s: src/network/SACAgent.cpp.s

.PHONY : src/network/SACAgent.s

# target to generate assembly for a file
src/network/SACAgent.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/SACAgent.cpp.s
.PHONY : src/network/SACAgent.cpp.s

src/network/ValueNetwork.o: src/network/ValueNetwork.cpp.o

.PHONY : src/network/ValueNetwork.o

# target to build an object file
src/network/ValueNetwork.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ValueNetwork.cpp.o
.PHONY : src/network/ValueNetwork.cpp.o

src/network/ValueNetwork.i: src/network/ValueNetwork.cpp.i

.PHONY : src/network/ValueNetwork.i

# target to preprocess a source file
src/network/ValueNetwork.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ValueNetwork.cpp.i
.PHONY : src/network/ValueNetwork.cpp.i

src/network/ValueNetwork.s: src/network/ValueNetwork.cpp.s

.PHONY : src/network/ValueNetwork.s

# target to generate assembly for a file
src/network/ValueNetwork.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/network/ValueNetwork.cpp.s
.PHONY : src/network/ValueNetwork.cpp.s

src/pid/PID.o: src/pid/PID.cpp.o

.PHONY : src/pid/PID.o

# target to build an object file
src/pid/PID.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/pid/PID.cpp.o
.PHONY : src/pid/PID.cpp.o

src/pid/PID.i: src/pid/PID.cpp.i

.PHONY : src/pid/PID.i

# target to preprocess a source file
src/pid/PID.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/pid/PID.cpp.i
.PHONY : src/pid/PID.cpp.i

src/pid/PID.s: src/pid/PID.cpp.s

.PHONY : src/pid/PID.s

# target to generate assembly for a file
src/pid/PID.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/pid/PID.cpp.s
.PHONY : src/pid/PID.cpp.s

src/servo/PCA9685.o: src/servo/PCA9685.cpp.o

.PHONY : src/servo/PCA9685.o

# target to build an object file
src/servo/PCA9685.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/PCA9685.cpp.o
.PHONY : src/servo/PCA9685.cpp.o

src/servo/PCA9685.i: src/servo/PCA9685.cpp.i

.PHONY : src/servo/PCA9685.i

# target to preprocess a source file
src/servo/PCA9685.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/PCA9685.cpp.i
.PHONY : src/servo/PCA9685.cpp.i

src/servo/PCA9685.s: src/servo/PCA9685.cpp.s

.PHONY : src/servo/PCA9685.s

# target to generate assembly for a file
src/servo/PCA9685.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/PCA9685.cpp.s
.PHONY : src/servo/PCA9685.cpp.s

src/servo/ServoKit.o: src/servo/ServoKit.cpp.o

.PHONY : src/servo/ServoKit.o

# target to build an object file
src/servo/ServoKit.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/ServoKit.cpp.o
.PHONY : src/servo/ServoKit.cpp.o

src/servo/ServoKit.i: src/servo/ServoKit.cpp.i

.PHONY : src/servo/ServoKit.i

# target to preprocess a source file
src/servo/ServoKit.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/ServoKit.cpp.i
.PHONY : src/servo/ServoKit.cpp.i

src/servo/ServoKit.s: src/servo/ServoKit.cpp.s

.PHONY : src/servo/ServoKit.s

# target to generate assembly for a file
src/servo/ServoKit.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/servo/ServoKit.cpp.s
.PHONY : src/servo/ServoKit.cpp.s

src/wire/Wire.o: src/wire/Wire.cpp.o

.PHONY : src/wire/Wire.o

# target to build an object file
src/wire/Wire.cpp.o:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/wire/Wire.cpp.o
.PHONY : src/wire/Wire.cpp.o

src/wire/Wire.i: src/wire/Wire.cpp.i

.PHONY : src/wire/Wire.i

# target to preprocess a source file
src/wire/Wire.cpp.i:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/wire/Wire.cpp.i
.PHONY : src/wire/Wire.cpp.i

src/wire/Wire.s: src/wire/Wire.cpp.s

.PHONY : src/wire/Wire.s

# target to generate assembly for a file
src/wire/Wire.cpp.s:
	$(MAKE) -f CMakeFiles/TATS.dir/build.make CMakeFiles/TATS.dir/src/wire/Wire.cpp.s
.PHONY : src/wire/Wire.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... TATS"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... src/detection/CascadeDetector.o"
	@echo "... src/detection/CascadeDetector.i"
	@echo "... src/detection/CascadeDetector.s"
	@echo "... src/detection/ObjectDetector.o"
	@echo "... src/detection/ObjectDetector.i"
	@echo "... src/detection/ObjectDetector.s"
	@echo "... src/detection/RCNNDetector.o"
	@echo "... src/detection/RCNNDetector.i"
	@echo "... src/detection/RCNNDetector.s"
	@echo "... src/detection/YoloDetector.o"
	@echo "... src/detection/YoloDetector.i"
	@echo "... src/detection/YoloDetector.s"
	@echo "... src/env/Env.o"
	@echo "... src/env/Env.i"
	@echo "... src/env/Env.s"
	@echo "... src/network/Normal.o"
	@echo "... src/network/Normal.i"
	@echo "... src/network/Normal.s"
	@echo "... src/network/PolicyNetwork.o"
	@echo "... src/network/PolicyNetwork.i"
	@echo "... src/network/PolicyNetwork.s"
	@echo "... src/network/QNetwork.o"
	@echo "... src/network/QNetwork.i"
	@echo "... src/network/QNetwork.s"
	@echo "... src/network/ReplayBuffer.o"
	@echo "... src/network/ReplayBuffer.i"
	@echo "... src/network/ReplayBuffer.s"
	@echo "... src/network/SACAgent.o"
	@echo "... src/network/SACAgent.i"
	@echo "... src/network/SACAgent.s"
	@echo "... src/network/ValueNetwork.o"
	@echo "... src/network/ValueNetwork.i"
	@echo "... src/network/ValueNetwork.s"
	@echo "... src/pid/PID.o"
	@echo "... src/pid/PID.i"
	@echo "... src/pid/PID.s"
	@echo "... src/servo/PCA9685.o"
	@echo "... src/servo/PCA9685.i"
	@echo "... src/servo/PCA9685.s"
	@echo "... src/servo/ServoKit.o"
	@echo "... src/servo/ServoKit.i"
	@echo "... src/servo/ServoKit.s"
	@echo "... src/wire/Wire.o"
	@echo "... src/wire/Wire.i"
	@echo "... src/wire/Wire.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

